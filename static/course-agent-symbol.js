const COURSE_AGENT_SYMBOL_STATES = new Set([
  "presence",
  "resonance",
  "flow",
  "bloom",
  "sustain",
  "error",
]);

class CourseAgentSymbol extends HTMLElement {
  static observedAttributes = ["state", "size", "animated", "aria-label"];

  connectedCallback() {
    if (!this.shadowRoot) this.#render();
    this.#sync();
  }

  attributeChangedCallback() {
    if (this.shadowRoot) this.#sync();
  }

  #sync() {
    const requestedState = this.getAttribute("state") || "presence";
    const state = COURSE_AGENT_SYMBOL_STATES.has(requestedState) ? requestedState : "presence";
    const requestedSize = Number(this.getAttribute("size"));
    const size = Number.isFinite(requestedSize) && requestedSize > 0 ? requestedSize : 48;
    this.dataset.state = state;
    this.style.setProperty("--course-agent-size", `${size}px`);

    const label = this.getAttribute("aria-label");
    if (label) {
      this.setAttribute("role", "img");
      this.removeAttribute("aria-hidden");
    } else {
      this.removeAttribute("role");
      this.setAttribute("aria-hidden", "true");
    }
  }

  #render() {
    this.attachShadow({ mode: "open" }).innerHTML = `
      <style>
        :host {
          --course-agent-deep-green: #0a3d34;
          --course-agent-primary-green: #00664f;
          --course-agent-green: #1e7a5e;
          --course-agent-light-green: #8ad26a;
          --course-agent-highlight: #d6e85a;
          display: inline-block;
          width: var(--course-agent-size, 48px);
          height: var(--course-agent-size, 48px);
          color: var(--course-agent-primary-green);
          flex: none;
          line-height: 0;
        }
        svg { width: 100%; height: 100%; overflow: visible; }
        .mark, .core, .halo, .ring, .energy, .tip {
          transform-box: fill-box;
          transform-origin: center;
        }
        .petal { fill: url(#course-agent-petal-gradient); }
        .core { fill: var(--course-agent-deep-green); }
        .core-aura { fill: url(#course-agent-core-aura); opacity: .62; }
        .halo { fill: var(--course-agent-highlight); opacity: 0; }
        .ring, .energy {
          fill: none;
          stroke: var(--course-agent-highlight);
          stroke-linecap: round;
          opacity: 0;
        }
        .ring { stroke-width: 1.3; transform-origin: 50px 85px; }
        .energy { stroke-width: 2.4; stroke-dasharray: .16 1; }
        .tip { fill: var(--course-agent-highlight); opacity: 0; }

        :host([animated][state="presence"]) .core,
        :host([animated][state="sustain"]) .core {
          animation: course-agent-breathe 3s ease-in-out infinite;
        }
        :host([animated][state="presence"]) .halo,
        :host([animated][state="sustain"]) .halo {
          animation: course-agent-halo 3s ease-in-out infinite;
        }
        :host([animated][state="resonance"]) .core {
          animation: course-agent-response .3s cubic-bezier(.4, 0, .2, 1) 1;
        }
        :host([animated][state="resonance"]) .ring {
          animation: course-agent-ripple .3s cubic-bezier(.4, 0, .2, 1) 1;
        }
        :host([animated][state="resonance"]) .ring.second { animation-delay: .06s; }
        :host([animated][state="flow"]) .energy {
          animation: course-agent-flow 1.5s linear infinite;
          animation-delay: var(--delay);
        }
        :host([animated][state="bloom"]) .mark {
          animation: course-agent-bloom-scale .48s ease-out 1;
        }
        :host([animated][state="bloom"]) .energy {
          stroke-dasharray: 1;
          animation: course-agent-bloom-path .48s ease-out 1;
        }
        :host([animated][state="bloom"]) .tip {
          animation: course-agent-tip .48s ease-out 1;
          animation-delay: var(--delay);
        }
        :host([animated][state="error"]) .core {
          animation: course-agent-error .45s ease-out 1;
        }

        @keyframes course-agent-breathe {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.03); opacity: .88; }
        }
        @keyframes course-agent-halo {
          0%, 100% { opacity: 0; transform: scale(.75); }
          50% { opacity: .18; transform: scale(1); }
        }
        @keyframes course-agent-response {
          0%, 100% { transform: scale(1); }
          45% { transform: scale(1.08); }
        }
        @keyframes course-agent-ripple {
          0% { opacity: .5; transform: scale(.45); }
          100% { opacity: 0; transform: scale(1.35); }
        }
        @keyframes course-agent-flow {
          0% { opacity: 0; stroke-dashoffset: 1.08; }
          15%, 72% { opacity: .82; }
          100% { opacity: 0; stroke-dashoffset: -.15; }
        }
        @keyframes course-agent-bloom-scale {
          0%, 100% { transform: scale(1); }
          52% { transform: scale(1.04); }
        }
        @keyframes course-agent-bloom-path {
          0% { opacity: 0; stroke-dashoffset: 1; }
          18% { opacity: .9; }
          75% { opacity: .65; stroke-dashoffset: 0; }
          100% { opacity: 0; stroke-dashoffset: 0; }
        }
        @keyframes course-agent-tip {
          0%, 35%, 100% { opacity: 0; transform: scale(.7); }
          70% { opacity: .72; transform: scale(1); }
        }
        @keyframes course-agent-error {
          0%, 100% { opacity: 1; transform: scale(1); }
          42% { opacity: .28; transform: scale(.94); }
          58% { opacity: .72; transform: scale(.98); }
        }

        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation: none !important;
          }
          :host([animated][state="flow"]) .energy,
          :host([animated][state="bloom"]) .energy {
            opacity: .18;
            stroke-dasharray: none;
            stroke-dashoffset: 0;
          }
          :host([animated][state="presence"]) .halo,
          :host([animated][state="sustain"]) .halo { opacity: .06; }
          :host([animated][state="resonance"]) .halo,
          :host([animated][state="resonance"]) .ring { opacity: .14; }
          :host([animated][state="bloom"]) .tip { opacity: .32; }
          :host([animated][state="error"]) .core { opacity: .55; }
        }
      </style>
      <svg viewBox="0 0 100 100" aria-hidden="true" focusable="false">
        <defs>
          <linearGradient id="course-agent-petal-gradient" x1="10" y1="78" x2="86" y2="12" gradientUnits="userSpaceOnUse">
            <stop offset="0" stop-color="#0a3d34"/>
            <stop offset=".32" stop-color="#00664f"/>
            <stop offset=".62" stop-color="#1e7a5e"/>
            <stop offset=".84" stop-color="#8ad26a"/>
            <stop offset="1" stop-color="#d6e85a"/>
          </linearGradient>
          <radialGradient id="course-agent-core-aura">
            <stop offset="0" stop-color="#d6e85a" stop-opacity=".72"/>
            <stop offset=".48" stop-color="#d6e85a" stop-opacity=".36"/>
            <stop offset="1" stop-color="#d6e85a" stop-opacity="0"/>
          </radialGradient>
        </defs>
        <g class="mark">
          <path class="petal" d="M49 80C42 66 30 57 15 56C7 55 6 47 11 42C16 37 24 39 31 44C41 52 47 66 49 80Z"/>
          <path class="petal" d="M49.5 80C46 59 37 40 27 30C21 24 24 16 31 15C38 14 42 20 44 27C49 43 50 62 49.5 80Z"/>
          <path class="petal" d="M50 80C48 58 43 34 42 19C41 9 47 5 50 5C53 5 59 9 58 19C57 34 52 58 50 80Z"/>
          <path class="petal" d="M50.5 80C50 62 51 43 56 27C58 20 62 14 69 15C76 16 79 24 73 30C63 40 54 59 50.5 80Z"/>
          <path class="petal" d="M51 80C53 66 59 52 69 44C76 39 84 37 89 42C94 47 93 55 85 56C70 57 58 66 51 80Z"/>
          <g class="energy-paths" pathLength="1">
            <path class="energy" pathLength="1" style="--delay:0s" d="M50 82Q40 64 16 50"/>
            <path class="energy" pathLength="1" style="--delay:.12s" d="M50 82Q45 48 31 21"/>
            <path class="energy" pathLength="1" style="--delay:.24s" d="M50 82Q50 44 50 10"/>
            <path class="energy" pathLength="1" style="--delay:.36s" d="M50 82Q55 48 69 21"/>
            <path class="energy" pathLength="1" style="--delay:.48s" d="M50 82Q60 64 84 50"/>
          </g>
          <circle class="tip" style="--delay:0s" cx="16" cy="50" r="2.4"/>
          <circle class="tip" style="--delay:.04s" cx="31" cy="21" r="2.4"/>
          <circle class="tip" style="--delay:.08s" cx="50" cy="10" r="2.4"/>
          <circle class="tip" style="--delay:.12s" cx="69" cy="21" r="2.4"/>
          <circle class="tip" style="--delay:.16s" cx="84" cy="50" r="2.4"/>
          <circle class="core-aura" cx="50" cy="85" r="11"/>
          <circle class="halo" cx="50" cy="85" r="12"/>
          <circle class="ring" cx="50" cy="85" r="10"/>
          <circle class="ring second" cx="50" cy="85" r="15"/>
          <circle class="core" cx="50" cy="85" r="5.5"/>
        </g>
      </svg>`;
  }
}

customElements.define("course-agent-symbol", CourseAgentSymbol);
window.CourseAgentSymbol = CourseAgentSymbol;
