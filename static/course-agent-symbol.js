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
        .mark, .petal, .growth-petal, .core, .halo, .ring, .energy, .tip {
          transform-box: fill-box;
          transform-origin: center;
        }
        .petal { fill: url(#course-agent-petal-gradient); }
        .growth-petal {
          fill: url(#course-agent-petal-gradient);
          opacity: 0;
          transform-box: view-box;
          transform-origin: 50px 92px;
        }
        .core { fill: var(--course-agent-deep-green); }
        .core-aura { fill: url(#course-agent-core-aura); opacity: .62; }
        .halo { fill: var(--course-agent-highlight); opacity: 0; }
        .ring, .energy {
          fill: none;
          stroke: var(--course-agent-highlight);
          stroke-linecap: round;
          opacity: 0;
        }
        .ring { stroke-width: 1.3; transform-origin: 50px 92px; }
        .energy { stroke-width: 2.4; stroke-dasharray: .16 1; }
        .tip { fill: var(--course-agent-highlight); opacity: 0; }

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
        :host([animated][state="flow"]) .petal { opacity: .14; }
        :host([animated][state="flow"]) .growth-petal {
          animation: course-agent-grow 1.5s cubic-bezier(.4, 0, .2, 1) infinite both;
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
        @keyframes course-agent-grow {
          0%, 12% { transform: scale(.02); opacity: 0; }
          22% { opacity: .88; }
          62%, 82% { transform: scale(1); opacity: 1; }
          100% { transform: scale(1); opacity: 0; }
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
          :host([animated][state="flow"]) .petal { opacity: 1; }
          :host([animated][state="flow"]) .growth-petal { opacity: 0; }
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
          <path id="course-agent-petal-1" d="M49.2 83C46 71 41 61 33 53C26 46 20 44 13 45C6 46 3 41 5 35C7 28 14 25 21 27C31 30 39 40 44 52C47 61 48.5 73 49.2 83Z"/>
          <path id="course-agent-petal-2" d="M49.6 83C48 67 45 51 39 38C35 31 29 27 24 25C18 22 17 16 21 12C25 7 33 8 37 12C43 18 45 26 47 36C50 52 50 69 49.6 83Z"/>
          <path id="course-agent-petal-3" d="M50 83C50 64 48 44 44 26C42 18 39 12 41 7C43 2 47 1 50 1C53 1 57 2 59 7C61 12 58 18 56 26C52 44 50 64 50 83Z"/>
          <path id="course-agent-petal-4" d="M50.4 83C50 69 50 52 53 36C55 26 57 18 63 12C67 8 75 7 79 12C83 16 82 22 76 25C71 27 65 31 61 38C55 51 52 67 50.4 83Z"/>
          <path id="course-agent-petal-5" d="M50.8 83C51.5 73 53 61 56 52C61 40 69 30 79 27C86 25 93 28 95 35C97 41 94 46 87 45C80 44 74 46 67 53C59 61 54 71 50.8 83Z"/>
          <mask id="course-agent-fan-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">
            <rect width="100" height="100" fill="white"/>
            <path d="M49.35 83C47 61 42 43 34 29M49.75 83C49 55 46 30 42 8M50.25 83C51 55 54 30 58 8M50.65 83C53 61 58 43 66 29" fill="none" stroke="black" stroke-width="2.4" stroke-linecap="round"/>
          </mask>
        </defs>
        <g class="mark">
          <g class="petals" mask="url(#course-agent-fan-mask)">
            <use class="petal" href="#course-agent-petal-1"/>
            <use class="petal" href="#course-agent-petal-2"/>
            <use class="petal" href="#course-agent-petal-3"/>
            <use class="petal" href="#course-agent-petal-4"/>
            <use class="petal" href="#course-agent-petal-5"/>
          </g>
          <g class="growth-petals" mask="url(#course-agent-fan-mask)">
            <use class="growth-petal" style="--delay:.24s" href="#course-agent-petal-1"/>
            <use class="growth-petal" style="--delay:.08s" href="#course-agent-petal-2"/>
            <use class="growth-petal" style="--delay:0s" href="#course-agent-petal-3"/>
            <use class="growth-petal" style="--delay:.16s" href="#course-agent-petal-4"/>
            <use class="growth-petal" style="--delay:.32s" href="#course-agent-petal-5"/>
          </g>
          <g class="energy-paths" pathLength="1">
            <path class="energy" pathLength="1" style="--delay:.24s" d="M50 92Q39 65 12 39"/>
            <path class="energy" pathLength="1" style="--delay:.08s" d="M50 92Q45 48 29 16"/>
            <path class="energy" pathLength="1" style="--delay:0s" d="M50 92Q50 44 50 7"/>
            <path class="energy" pathLength="1" style="--delay:.16s" d="M50 92Q55 48 71 16"/>
            <path class="energy" pathLength="1" style="--delay:.32s" d="M50 92Q61 65 88 39"/>
          </g>
          <circle class="tip" style="--delay:.12s" cx="12" cy="39" r="2.2"/>
          <circle class="tip" style="--delay:.04s" cx="29" cy="16" r="2.2"/>
          <circle class="tip" style="--delay:0s" cx="50" cy="7" r="2.2"/>
          <circle class="tip" style="--delay:.08s" cx="71" cy="16" r="2.2"/>
          <circle class="tip" style="--delay:.16s" cx="88" cy="39" r="2.2"/>
          <circle class="core-aura" cx="50" cy="92" r="9"/>
          <circle class="halo" cx="50" cy="92" r="11"/>
          <circle class="ring" cx="50" cy="92" r="9"/>
          <circle class="ring second" cx="50" cy="92" r="14"/>
          <circle class="core" cx="50" cy="92" r="4.8"/>
        </g>
      </svg>`;
  }
}

customElements.define("course-agent-symbol", CourseAgentSymbol);
window.CourseAgentSymbol = CourseAgentSymbol;
