/**
 * IIT Guwahati - Multimodal Broadcast Analytics System RAG Chatbot
 * Interactive Scripts & Mobile Optimizations (script.js)
 */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initReadingProgressBar();
  initMobileMenu();
  initArchitectureExplorer();
  initCodeTabs();
  initCopyCodeButtons();
  initBenchmarkAnimations();
  initScrollSpy();
  initFloatingActions();
  initMathScrollOptimization();
});

/* --------------------------------------------------------------------------
   1. Theme Toggle (Dark / Light Mode)
   -------------------------------------------------------------------------- */
function initThemeToggle() {
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const themeIcon = document.getElementById('theme-icon');
  
  // Check local storage or system preference
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initialTheme = savedTheme ? savedTheme : (prefersDark ? 'dark' : 'light');

  setTheme(initialTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
      localStorage.setItem('theme', newTheme);
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeIcon) {
      if (theme === 'dark') {
        themeIcon.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="5"></circle>
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
          </svg>`;
      } else {
        themeIcon.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>`;
      }
    }
  }
}

/* --------------------------------------------------------------------------
   2. Reading Progress Bar
   -------------------------------------------------------------------------- */
function initReadingProgressBar() {
  const progressBar = document.getElementById('progress-bar');
  if (!progressBar) return;

  window.addEventListener('scroll', () => {
    const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = totalHeight > 0 ? (window.pageYOffset / totalHeight) * 100 : 0;
    progressBar.style.width = `${progress}%`;
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   3. Mobile Menu Toggle, Overlay & Body Lock
   -------------------------------------------------------------------------- */
function initMobileMenu() {
  const mobileToggleBtn = document.getElementById('mobile-toggle-btn');
  const navMenu = document.getElementById('nav-menu');
  const overlay = document.getElementById('mobile-menu-overlay');

  if (!mobileToggleBtn || !navMenu) return;

  function toggleMenu(isOpen) {
    const shouldOpen = isOpen !== undefined ? isOpen : !navMenu.classList.contains('mobile-open');
    
    if (shouldOpen) {
      navMenu.classList.add('mobile-open');
      mobileToggleBtn.classList.add('open');
      mobileToggleBtn.setAttribute('aria-expanded', 'true');
      if (overlay) overlay.classList.add('active');
      document.body.classList.add('menu-locked');
    } else {
      navMenu.classList.remove('mobile-open');
      mobileToggleBtn.classList.remove('open');
      mobileToggleBtn.setAttribute('aria-expanded', 'false');
      if (overlay) overlay.classList.remove('active');
      document.body.classList.remove('menu-locked');
    }
  }

  mobileToggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleMenu();
  });

  if (overlay) {
    overlay.addEventListener('click', () => {
      toggleMenu(false);
    });
  }

  // Close menu when clicking any nav link
  const navLinks = navMenu.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      toggleMenu(false);
    });
  });

  // Close menu on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && navMenu.classList.contains('mobile-open')) {
      toggleMenu(false);
    }
  });

  // Close menu on resize back to desktop
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768 && navMenu.classList.contains('mobile-open')) {
      toggleMenu(false);
    }
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   4. Interactive Architecture Explorer
   -------------------------------------------------------------------------- */
const architectureSteps = [
  {
    step: 1,
    id: "step-ingest",
    nodeId: "node-ingest",
    title: "1. Document Ingestion & PyMuPDF Parsing",
    desc: "Accepts multimodal academic papers and broadcast documents (PDF, TXT, MD). Uses PyMuPDF (fitz) for page-level text extraction with exact page labeling and metadata serialization into results/processed_documents.pkl.",
    tech: "PyMuPDF (fitz), Python Pickle",
    output: "LangChain Document objects with page metadata",
    metric: "< 250ms parse time per 20-page document"
  },
  {
    step: 2,
    id: "step-chunk",
    nodeId: "node-chunk",
    title: "2. Recursive Chunking & FAISS Vector Indexing",
    desc: "Splits documents using RecursiveCharacterTextSplitter (chunk_size=500, overlap=100) with hierarchical separators. Generates 384-dimensional dense vectors using all-MiniLM-L6-v2 and stores them in an in-memory FAISS L2/Cosine index.",
    tech: "all-MiniLM-L6-v2 (CPU), FAISS Vector Store",
    output: "Dense 384-d FAISS index + chunk ID mappings",
    metric: "500 chars/chunk, 100 chars semantic overlap"
  },
  {
    step: 3,
    id: "step-guardrail",
    nodeId: "node-guardrail",
    title: "3. Query Guardrail Screening Filter",
    desc: "Pre-screen incoming queries with zero-shot LLM categorization to strictly filter out prompt injections, jailbreaks, and off-topic questions, protecting LLM compute while allowing research & graph requests.",
    tech: "LangChain Prompt Filter + Groq LLM",
    output: "ALLOWED or BLOCKED classification decision",
    metric: "0 ms vector cost on filtered malicious queries"
  },
  {
    step: 4,
    id: "step-multiquery",
    nodeId: "node-multiquery",
    title: "4. Question Condensation & Multi-Query Expansion",
    desc: "Resolves conversational pronouns (it, this, that) from history to generate a standalone query. Then prompts the LLM to generate 3 alternative query variations across technical terms and sub-questions, executing parallel retrieval across FAISS.",
    tech: "MultiQueryRetriever, ChatGroq / Ollama",
    output: "3 unique query variations + union deduplication",
    metric: "+34.5% recall boost over naive vector search"
  },
  {
    step: 5,
    id: "step-rerank",
    nodeId: "node-rerank",
    title: "5. TinyBERT Cross-Encoder Reranking",
    desc: "Pairs the standalone query with all deduplicated candidate chunks and passes them through cross-encoder/ms-marco-TinyBERT-L-2-v2. Evaluates query-document interaction simultaneously to produce calibrated relevance logits, selecting the Top-5 chunks.",
    tech: "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    output: "Top 5 relevance-sorted context chunks",
    metric: "MRR@5 jumps from 0.58 to 0.92"
  },
  {
    step: 6,
    id: "step-generation",
    nodeId: "node-generation",
    title: "6. Grounded LLM Generation & Graph Plotter",
    desc: "Injects Top-5 reranked context into a strictly grounded prompt (saying 'I don't know' if unverified). If visual data is requested, extracts validated JSON and plots charts via Matplotlib. Passes text through a custom KaTeX LaTeX mathematical formatter.",
    tech: "Groq (GPT-OSS-120B / Qwen) + Matplotlib + KaTeX",
    output: "Markdown stream + LaTeX equations + Matplotlib charts",
    metric: "Zero hallucination on missing facts, LaTeX auto-repair"
  }
];

function initArchitectureExplorer() {
  const stepButtons = document.querySelectorAll('.pipeline-step-btn');
  const svgNodes = document.querySelectorAll('.arch-node');

  const detailNumber = document.getElementById('step-detail-num');
  const detailTitle = document.getElementById('step-detail-title');
  const detailDesc = document.getElementById('step-detail-desc');
  const detailTech = document.getElementById('step-detail-tech');
  const detailOutput = document.getElementById('step-detail-output');
  const detailMetric = document.getElementById('step-detail-metric');
  const stepNav = document.getElementById('pipeline-steps-nav');

  function updateActiveStep(stepIndex) {
    const stepData = architectureSteps[stepIndex];
    if (!stepData) return;

    // Update buttons
    stepButtons.forEach((btn, idx) => {
      const isActive = idx === stepIndex;
      btn.classList.toggle('active', isActive);
      if (isActive && stepNav) {
        // Scroll button into view smoothly on mobile horizontally
        btn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    });

    // Update SVG nodes
    svgNodes.forEach((node) => {
      const isTarget = node.id === stepData.nodeId;
      node.classList.toggle('active', isTarget);
    });

    // Update details card
    if (detailNumber) detailNumber.textContent = stepData.step;
    if (detailTitle) detailTitle.textContent = stepData.title;
    if (detailDesc) detailDesc.textContent = stepData.desc;
    if (detailTech) detailTech.textContent = stepData.tech;
    if (detailOutput) detailOutput.textContent = stepData.output;
    if (detailMetric) detailMetric.textContent = stepData.metric;
  }

  stepButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
      updateActiveStep(index);
    });
  });

  svgNodes.forEach((node) => {
    node.addEventListener('click', () => {
      const stepIndex = architectureSteps.findIndex(s => s.nodeId === node.id);
      if (stepIndex !== -1) {
        updateActiveStep(stepIndex);
      }
    });
  });

  // Default initialize with step 0
  updateActiveStep(0);
}

/* --------------------------------------------------------------------------
   5. Code Showcase Tabs
   -------------------------------------------------------------------------- */
function initCodeTabs() {
  const tabButtons = document.querySelectorAll('.code-tab-btn');
  const tabPanels = document.querySelectorAll('.code-content-panel');
  const tabsHeader = document.querySelector('.code-tabs-header');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const activePanel = document.getElementById(targetId);
      if (activePanel) {
        activePanel.classList.add('active');
      }

      if (tabsHeader) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    });
  });
}

/* --------------------------------------------------------------------------
   6. Copy Code Buttons & Toast Feedback
   -------------------------------------------------------------------------- */
function initCopyCodeButtons() {
  const copyButtons = document.querySelectorAll('.copy-code-btn');
  const toast = document.getElementById('toast');

  copyButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const panel = btn.closest('.code-content-panel');
      const codeElement = panel ? panel.querySelector('code') : null;
      if (!codeElement) return;

      const codeText = codeElement.innerText || codeElement.textContent;
      navigator.clipboard.writeText(codeText).then(() => {
        showToast('Code copied to clipboard!');
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg> Copied!`;
        setTimeout(() => {
          btn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg> Copy`;
        }, 2000);
      });
    });
  });

  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }
}

/* --------------------------------------------------------------------------
   7. Benchmark Bars Animation on Scroll
   -------------------------------------------------------------------------- */
function initBenchmarkAnimations() {
  const bars = document.querySelectorAll('.bar-fill');
  if (!bars.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const fill = entry.target;
        const targetWidth = fill.getAttribute('data-width') || '0%';
        fill.style.width = targetWidth;
        observer.unobserve(fill);
      }
    });
  }, { threshold: 0.15 });

  bars.forEach(bar => {
    bar.style.width = '0%';
    observer.observe(bar);
  });
}

/* --------------------------------------------------------------------------
   8. ScrollSpy Navigation
   -------------------------------------------------------------------------- */
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id], header[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.pageYOffset + 120;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   9. Floating Action Controls (Back to Top & Live App)
   -------------------------------------------------------------------------- */
function initFloatingActions() {
  const floatingActions = document.getElementById('floating-actions');
  const floatingTopBtn = document.getElementById('floating-top-btn');

  if (!floatingActions) return;

  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 320) {
      floatingActions.classList.add('show');
    } else {
      floatingActions.classList.remove('show');
    }
  }, { passive: true });

  if (floatingTopBtn) {
    floatingTopBtn.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }
}

/* --------------------------------------------------------------------------
   10. Math Formula Scroll Hint & Mobile Interaction
   -------------------------------------------------------------------------- */
function initMathScrollOptimization() {
  const mathBoxes = document.querySelectorAll('.math-formula-box');
  mathBoxes.forEach(box => {
    box.addEventListener('touchstart', () => {}, { passive: true });
  });
}
