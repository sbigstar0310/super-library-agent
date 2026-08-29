export default [
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/build/**",
      "**/.next/**",
      "**/.turbo/**",
      "**/.vite/**",
      "**/coverage/**",
      "**/eval_results/**",
      "**/embeddings/**",
    ],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      complexity: ["warn", 0],
    },
  },
];
