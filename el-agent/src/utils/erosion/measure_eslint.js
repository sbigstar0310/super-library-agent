/*
 * measure_eslint.js
 *
 * Walks a directory and reports per-function cyclomatic complexity (CC)
 * and source lines of code (SLOC) for JS/JSX/TS/TSX files.
 *
 * Scope: JavaScript family only. Python codebases are measured by SCB's
 * `slop_code.metrics` pipeline (called from scripts/metrics/scb_quality.py for
 * tasks in PY_TASKS); this script is invoked for webgen.
 *
 * Hybrid design:
 *   - ESLint `complexity` rule (max=0) is the canonical CC source.
 *     v9 reports only the function-identifier/arrow span, not the body span,
 *     so its `endLine` is not usable for SLOC.
 *   - @babel/parser is used for accurate function-span (line, endLine) and
 *     name resolution (variable-bound arrow funcs, object methods, etc.).
 *   - Per file, walk Babel functions in source order and pair them by
 *     `loc.start.line` with ESLint complexity messages on the same file
 *     (both engines agree on start-line). Multiple functions starting on
 *     the same line are paired in declaration order.
 *
 * Usage:
 *   node measure_eslint.js <app_dir>
 *
 * Output: JSON array on stdout
 *   [{ file, name, line, endLine, sloc, cc }, ...]
 */

const fs = require("fs");
const path = require("path");
const { ESLint } = require("eslint");
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;

const APP_DIR = process.argv[2];
if (!APP_DIR) {
  console.error("Usage: node measure_eslint.js <app_dir>");
  process.exit(1);
}

const CONFIG_FILE = path.join(__dirname, "eslint.config.mjs");

const BABEL_PLUGINS = [
  "jsx",
  "typescript",
  "classProperties",
  "classPrivateProperties",
  "classPrivateMethods",
  "decorators-legacy",
  "dynamicImport",
  "optionalChaining",
  "nullishCoalescingOperator",
  "topLevelAwait",
];

function parseFile(source) {
  return parser.parse(source, {
    sourceType: "module",
    allowReturnOutsideFunction: true,
    allowImportExportEverywhere: true,
    errorRecovery: true,
    plugins: BABEL_PLUGINS,
  });
}

function getFunctionName(fnPath) {
  const node = fnPath.node;

  if (node.id && node.id.name) return node.id.name;

  if (node.type === "ClassMethod" || node.type === "ClassPrivateMethod") {
    if (node.key) {
      if (node.key.name) return node.key.name;
      if (node.key.type === "StringLiteral") return node.key.value;
    }
    return "<method>";
  }

  if (node.type === "ObjectMethod") {
    if (node.key && node.key.name) return node.key.name;
    return "<method>";
  }

  const parent = fnPath.parent;
  if (parent) {
    if (parent.type === "VariableDeclarator" && parent.id && parent.id.name) {
      return parent.id.name;
    }
    if (parent.type === "AssignmentExpression" && parent.left) {
      if (parent.left.type === "Identifier") return parent.left.name;
      if (
        parent.left.type === "MemberExpression" &&
        parent.left.property &&
        parent.left.property.name
      ) {
        return parent.left.property.name;
      }
    }
    if (parent.type === "ObjectProperty" && parent.key) {
      if (parent.key.name) return parent.key.name;
      if (parent.key.type === "StringLiteral") return parent.key.value;
    }
  }

  return "<anonymous>";
}

function collectBabelFunctions(filePath) {
  // Returns [{ startLine, endLine, name }, ...] in source order.
  let source;
  try {
    source = fs.readFileSync(filePath, "utf8");
  } catch (_) {
    return null;
  }
  let ast;
  try {
    ast = parseFile(source);
  } catch (_) {
    return null;
  }
  const fns = [];
  traverse(ast, {
    Function(fnPath) {
      const loc = fnPath.node.loc;
      if (!loc) return;
      fns.push({
        startLine: loc.start.line,
        endLine: loc.end.line,
        name: getFunctionName(fnPath),
      });
    },
  });
  return fns;
}

function collectEslintCC(messages) {
  // Filter and parse complexity messages → [{ line, cc }, ...] in report order.
  // ESLint v9 reports identifier/arrow location with line == startLine of the
  // function as defined by Babel, so `line` alone is a reliable join key.
  const out = [];
  for (const m of messages) {
    if (m.ruleId !== "complexity") continue;
    const match = m.message.match(/has a complexity of (\d+)/);
    if (!match) continue;
    out.push({ line: m.line, cc: parseInt(match[1], 10) });
  }
  return out;
}

function pairByStartLine(babelFns, eslintMsgs) {
  // Group ESLint messages by line, then consume in declaration order per line.
  const queues = new Map();
  for (const m of eslintMsgs) {
    if (!queues.has(m.line)) queues.set(m.line, []);
    queues.get(m.line).push(m.cc);
  }
  const paired = [];
  for (const fn of babelFns) {
    const q = queues.get(fn.startLine);
    if (q && q.length > 0) {
      const cc = q.shift();
      paired.push({ ...fn, cc });
    } else {
      // Babel saw a function ESLint didn't report on — e.g., parser disagreement
      // on a synthetic node. Skip rather than fabricate CC.
    }
  }
  return paired;
}

async function main() {
  const absAppDir = path.resolve(APP_DIR);
  if (!fs.existsSync(absAppDir)) {
    console.error(`app_dir not found: ${absAppDir}`);
    process.exit(1);
  }

  const eslint = new ESLint({
    overrideConfigFile: CONFIG_FILE,
    cwd: absAppDir,
    errorOnUnmatchedPattern: false,
  });

  const results = await eslint.lintFiles(["**/*.{js,jsx,ts,tsx}"]);

  const out = [];
  const parseErrors = [];

  for (const r of results) {
    const rel = path.relative(absAppDir, r.filePath);

    for (const m of r.messages) {
      if (m.fatal) {
        parseErrors.push({ file: rel, error: `eslint: ${m.message}` });
        break;
      }
    }

    const babelFns = collectBabelFunctions(r.filePath);
    if (babelFns === null) {
      parseErrors.push({ file: rel, error: "babel parse failed" });
      continue;
    }

    const ccs = collectEslintCC(r.messages);
    const paired = pairByStartLine(babelFns, ccs);

    for (const p of paired) {
      out.push({
        file: rel,
        name: p.name,
        line: p.startLine,
        endLine: p.endLine,
        sloc: p.endLine - p.startLine + 1,
        cc: p.cc,
      });
    }
  }

  if (parseErrors.length > 0) {
    process.stderr.write(
      `measure_eslint: ${parseErrors.length} file(s) had issues\n`
    );
    for (const e of parseErrors.slice(0, 5)) {
      process.stderr.write(`  ${e.file}: ${e.error}\n`);
    }
  }

  process.stdout.write(JSON.stringify(out));
}

main().catch((e) => {
  process.stderr.write(`measure_eslint failed: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
});
