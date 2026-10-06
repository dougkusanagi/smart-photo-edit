module.exports = {
    darkMode: 'class',
    theme: { extend: {
      fontFamily: { sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', 'system-ui', '"Segoe UI"', 'Roboto', 'sans-serif'] },
      colors: {
        bg:      'rgb(var(--bg) / <alpha-value>)',
        canvas:  'rgb(var(--canvas) / <alpha-value>)',
        glass:   'rgb(var(--glass) / <alpha-value>)',
        ink:     'rgb(var(--ink) / <alpha-value>)',
        mute:    'rgb(var(--mute) / <alpha-value>)',
        faint:   'rgb(var(--faint) / <alpha-value>)',
        line:    'rgb(var(--line) / <alpha-value>)',
        accent:  'rgb(var(--accent) / <alpha-value>)',
        danger:  'rgb(var(--danger) / <alpha-value>)',
        ok:      'rgb(var(--ok) / <alpha-value>)',
      },
      transitionTimingFunction: { out: 'cubic-bezier(.32,.72,0,1)' },
    } },
  };
module.exports.content = ["./smart_photo_edit/web/index.html"];
