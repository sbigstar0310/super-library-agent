/* Super Library Agent — project page.
   No dependencies. Progressive: everything below degrades to a static page. */

(function () {
  'use strict';

  /* ---------- tiny JS/JSX highlighter ----------------------------------
     Deliberately small: the code shown is plain modern JS/JSX, and a real
     highlighter would outweigh the page. Tokenizes in one pass so nothing
     can be highlighted inside a string or comment. */

  var KEYWORDS = new RegExp('\\b(' + [
    'import', 'from', 'export', 'default', 'const', 'let', 'var', 'function',
    'return', 'if', 'else', 'for', 'while', 'try', 'catch', 'finally', 'throw',
    'new', 'typeof', 'instanceof', 'delete', 'in', 'of', 'class', 'extends',
    'async', 'await', 'yield', 'this', 'true', 'false', 'null', 'undefined'
  ].join('|') + ')\\b');

  var RULES = [
    ['com', /^\/\*[\s\S]*?\*\/|^\/\/[^\n]*/],
    ['str', /^`(?:\\[\s\S]|[^`\\])*`|^'(?:\\[\s\S]|[^'\\\n])*'|^"(?:\\[\s\S]|[^"\\\n])*"/],
    ['tag', /^<\/?[A-Za-z][\w.]*|^\/?>/],
    ['num', /^\b\d[\w.]*/],
    ['key', KEYWORDS]
  ];

  // characters that can begin a token: comment/string/JSX/number/identifier
  var PLAIN = /[/'"`<\d]|[A-Za-z_$]/g;

  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;';
    });
  }

  function highlight(src) {
    var out = '';
    var i = 0;
    while (i < src.length) {
      var rest = src.slice(i);
      var hit = null;
      for (var r = 0; r < RULES.length; r++) {
        var m = RULES[r][1].exec(rest);
        if (m && m.index === 0) { hit = [RULES[r][0], m[0]]; break; }
      }
      if (hit) {
        out += '<span class="t-' + hit[0] + '">' + esc(hit[1]) + '</span>';
        i += hit[1].length;
      } else {
        // Consume a run of plain characters up to the next character that could
        // start a token. Anchored at index 1 so we always advance by >= 1, and
        // deliberately NOT using \b — on a freshly sliced string \b matches at
        // index 0 for any word character, which collapsed this to one character
        // per iteration and made highlighting quadratic.
        PLAIN.lastIndex = 1;
        var m2 = PLAIN.exec(rest);
        var step = m2 ? m2.index : rest.length;
        out += esc(rest.slice(0, step));
        i += step;
      }
    }
    return out;
  }

  /* ---------- library browser ---------- */

  var GROUPS = [
    ['components', 'Components'],
    ['hooks', 'Hooks'],
    ['utils', 'Utilities'],
    ['styles', 'Stylesheets'],
    ['barrel', 'Entry point']
  ];

  function reuseBadge(file) {
    if (file.reuseApps === null || file.reuseApps === undefined) return '';
    var cls = file.reuseApps === 0 ? 'reuse zero' : 'reuse';
    return '<span class="' + cls + '">' + esc(String(file.reuseApps)) +
      '/8 apps</span>';
  }

  function renderLibrary(lib) {
    var tree = document.getElementById('tree');
    if (!tree) return;
    var buttons = [];

    GROUPS.forEach(function (g) {
      var files = lib.files.filter(function (f) { return f.kind === g[0]; });
      if (!files.length) return;
      var head = document.createElement('div');
      head.className = 'tree-group';
      head.textContent = g[1];
      tree.appendChild(head);

      files.forEach(function (f) {
        var b = document.createElement('button');
        b.type = 'button';
        b.setAttribute('role', 'tab');
        b.setAttribute('aria-selected', 'false');
        b.innerHTML = '<span class="fname">' +
          esc(f.path.split('/').pop()) + '</span>' + reuseBadge(f);
        b.addEventListener('click', function () { select(f, b); });
        tree.appendChild(b);
        buttons.push(b);
      });
    });

    function select(file, btn) {
      buttons.forEach(function (b) { b.setAttribute('aria-selected', 'false'); });
      btn.setAttribute('aria-selected', 'true');
      document.getElementById('viewer-path').textContent = 'lib/src/' + file.path;
      document.getElementById('viewer-lines').textContent = file.lines + ' lines';
      document.getElementById('viewer-symbols').textContent = file.symbols.length
        ? file.symbols.map(function (s) {
            return s.name + ' · ' + s.reuseApps + '/8';
          }).join('   ')
        : file.owner
          ? 'styles for ' + file.owner
          : 're-exports every public symbol';
      document.getElementById('viewer-code').innerHTML = highlight(file.code);
    }

    if (buttons.length) select(lib.files[0], buttons[0]);

    var src = lib.source;
    var note = document.getElementById('lib-source');
    if (note) {
      note.textContent = 'One of the extracted libraries: WebGen-Bench suite ' +
        src.suite + ', trial ' + src.trial + ', after round ' + src.round + '.';
    }
  }

  /* ---------- migration diff ---------- */

  function renderDiff(mig) {
    var before = document.getElementById('ba-before');
    if (!before) return;

    var drop = Math.round(100 * (mig.before.lines - mig.after.lines) / mig.before.lines);
    document.getElementById('diff-file').textContent = mig.file;
    document.getElementById('diff-meta').innerHTML =
      '<b>' + mig.before.lines + ' &rarr; ' + mig.after.lines + ' lines</b>' +
      '<span class="drop">&minus;' + drop + '%</span>';

    document.getElementById('ba-before-label').textContent =
      'before  ·  ' + mig.before.lines + ' lines';
    document.getElementById('ba-after-label').textContent =
      'after  ·  ' + mig.after.lines + ' lines';
    function render(rows) {
      return rows.map(function (r) {
        if (r.t === 'fold') {
          var what = r.kind === 'del' ? 'lines removed'
                   : r.kind === 'add' ? 'lines added' : 'unchanged lines';
          return '<span class="ln fold">&hellip; ' + r.hidden + ' ' + what +
            '</span>';
        }
        return '<span class="ln ' + r.t + '"><i>' + r.n + '</i>' +
          highlight(r.s) + '</span>';
      }).join('');
    }
    before.innerHTML = render(mig.before.folded);
    document.getElementById('ba-after').innerHTML = render(mig.after.folded);
  }

/* ---------- pipeline animation ---------- */

  function initPipeline() {
    var fig = document.getElementById('sla-anim');
    if (!fig) return;
    var steps = Array.prototype.slice.call(
      document.querySelectorAll('.anim-steps li'));
    if (!steps.length) return;

    var n = steps.length;
    var i = 0;
    var timer = null;
    var reduced = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    // steps carry very different amounts of change; dwell accordingly
    var DWELL = [2200, 3400, 2800, 3200, 4800];

    function show(k) {
      i = ((k % n) + n) % n;
      fig.setAttribute('data-step', String(i + 1));
      steps.forEach(function (li, j) {
        li.classList.toggle('on', j === i);
        li.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      });
      steps[i].style.setProperty('--dwell', DWELL[i] + 'ms');
    }
    function restartBar() {
      // re-trigger the dwell-progress animation from zero
      var li = steps[i];
      li.classList.remove('on');
      void li.offsetWidth;
      li.classList.add('on');
    }
    function tick() {
      timer = setTimeout(function () { show(i + 1); tick(); }, DWELL[i]);
    }
    function play() {
      if (reduced || timer) return;
      fig.classList.remove('anim-paused');
      restartBar();
      tick();
    }
    function pause() {
      fig.classList.add('anim-paused');
      if (timer) { clearTimeout(timer); timer = null; }
    }

    steps.forEach(function (li, j) {
      li.tabIndex = 0;
      li.setAttribute('role', 'button');
      li.addEventListener('click', function () { pause(); show(j); });
      li.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pause(); show(j); }
      });
    });
    fig.addEventListener('mouseenter', pause);
    fig.addEventListener('mouseleave', play);
    fig.addEventListener('focusin', pause);
    fig.addEventListener('focusout', function (e) {
      if (!fig.contains(e.relatedTarget) && !fig.matches(':hover')) play();
    });

    fig.classList.add('anim-paused');
    show(reduced ? n - 1 : 0);

    // only run while it is on screen
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) { e.isIntersecting ? play() : pause(); });
      }, {threshold: 0.25}).observe(fig);
    } else {
      play();
    }
  }

  /* ---------- benchmark tabs ---------- */

  function initTabs() {
    var tabs = document.querySelectorAll('.tabs [data-bench]');
    if (!tabs.length) return;
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        tabs.forEach(function (t) { t.setAttribute('aria-selected', 'false'); });
        tab.setAttribute('aria-selected', 'true');
        document.querySelectorAll('[data-bench-panel]').forEach(function (p) {
          p.hidden = p.getAttribute('data-bench-panel') !== tab.dataset.bench;
        });
      });
    });
  }

  /* ---------- scroll spy ---------- */

  function initSpy() {
    var links = Array.prototype.slice.call(
      document.querySelectorAll('.nav-links a'));
    var targets = links
      .map(function (a) { return document.querySelector(a.getAttribute('href')); })
      .filter(Boolean);
    if (!targets.length || !('IntersectionObserver' in window)) return;

    var seen = {};
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting; });
      for (var i = 0; i < targets.length; i++) {
        if (seen[targets[i].id]) {
          links.forEach(function (a) { a.classList.remove('active'); });
          links[i].classList.add('active');
          break;
        }
      }
    }, { rootMargin: '-72px 0px -70% 0px' });
    targets.forEach(function (t) { obs.observe(t); });
  }

  /* ---------- bibtex copy ---------- */

  function initCopy() {
    var btn = document.getElementById('copy-bib');
    var pre = document.getElementById('bibtex');
    if (!btn || !pre || !navigator.clipboard) return;
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText(pre.textContent).then(function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      });
    });
  }

  /* ---------- boot ---------- */

  function load(url, render, label) {
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function (err) {
        console.warn('could not render ' + label, err);
      });
  }

  initPipeline();
  initTabs();
  initSpy();
  initCopy();
  load('data/library.json', renderLibrary, 'library');
  load('data/migration.json', renderDiff, 'migration');
})();
