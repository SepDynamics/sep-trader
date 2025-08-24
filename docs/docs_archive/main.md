look at the todo.md and devise tasks outlined to deliver on that in combination wiht this new outline:

Of course. You're preparing to have a powerful AI model, Claude, act as a strategic partner to synthesize your entire project—code, docs, and vision—into a compelling narrative for a Series A fundraise.

To get the best results, you need to give Claude a clear, comprehensive, and structured set of instructions. Think of it as briefing a new co-founder who has instant, perfect recall.

Here is a master prompt you can provide to Claude. It's designed to guide it through a "clean sweep" and produce the deliverables you need to present your case.

Master Instructions for Claude Sonnet 4.1

Your Role: You are my strategic co-founder and lead technical writer for SEP Dynamics. Your expertise spans deep technology, business strategy, and investor communication.

My Objective: I am preparing for a $15M Series A fundraise. My goal is to perform a "clean sweep" of my project, SEP Dynamics, to create a crystal-clear, compelling, and unified narrative that justifies this investment.

Your Resources: You have full access to my GitHub repository (provided as sep_code_snapshot_20250812_225406.txt), my investor pitch deck (SEPPRESEEDV7.pptx), and all supplementary documentation (the .md files from my /docs directory).

Your Core Mandate: Your primary mission is to synthesize all provided information—from the philosophical underpinnings in BOOK1.md to the C++/CUDA code in the repo—into a single, cohesive, and powerful narrative. This narrative must be validated at every step by the provided source code and documentation.

Specific Tasks: The "Clean Sweep"

Please perform the following tasks, drawing connections between the pitch deck, the code, and the documentation.

I. Narrative & Storytelling

Synthesize the Grand Vision: Read BOOK1.md. Distill the core philosophy ("Reality as a Bounded Computational System," "Identity is Recursion," etc.) and connect it directly to the technology described in the pitch deck ("Quantum-Inspired Financial Intelligence"). Explain how the philosophical concepts are made real in the code.

Distill the Core Innovation (The "Weather Radar"): The pitch deck introduces "Quantum Field Harmonics" (QFH). Explain this concept in a simple, powerful way for an investor. Use the technical details from src/quantum/bitspace/qfh.h and the patent documents (01_QFH_INVENTION_DISCLOSURE.md) to describe what QFH, NULL_STATE, FLIP, and RUPTURE actually do at a practical level to predict market shifts.

Validate the Moat: The deck claims "Protected IP" with a patent pending. Cross-reference the "Key Claims Covered" on the IP slide with the code in src/quantum/bitspace/ and the patent disclosure documents. Confirm that the patented concepts (QFH, QBSA, Riemannian Optimization) are not just ideas, but are implemented in the C++/CUDA engine.

II. Technical & Product Deep Dive

Map Pitch Deck Claims to Code: For each technical claim in the pitch deck, find the specific code that proves it. Create a simple table:

Claim: "GPU-accelerated C++/CUDA engine" -> Evidence: src/apps/oanda_trader/CMakeLists.txt (enables CUDA), tick_cuda_kernels.cu, forward_window_kernels.cu.

Claim: "Hybrid Architecture (Local Training/Cloud Execution)" -> Evidence: Product & Platform Maturity slide details, CLI tools in src/cli suggest remote management.

Claim: "Enterprise-Ready Components (Docker, PostgreSQL, Nginx)" -> Evidence: Product & Platform Maturity slide, ENTERPRISE_DATA_LAYER.md.

Assess Product & Platform Maturity: Go beyond the pitch deck slide. Based on the code structure (e.g., src/cli, src/app, CMake structure, presence of docker-compose mentions), evaluate the true maturity of the platform. Is it a prototype or a robust, deployable system? Provide a brief assessment.

Clarify the "Quantum-Inspired" Label: Using POSITIONING_CLARIFICATION.md and your analysis, craft a clear, one-paragraph explanation for investors. It should convey the technological sophistication without overstating claims (i.e., clarifying it's not a quantum computer).

III. Strategy & Financials

Refine the Go-To-Market Strategy: The GTM slide has three phases. Flesh these out with concrete, credible steps based on the technology you've analyzed.

Phase 1 (Direct to Quants): How does the CLI (src/cli) and the professional architecture support this?

Phase 2 (Platform & API Expansion): How do the files in src/c_api and src/app prove you're ready for this?

Phase 3 (Mainstream Adoption): How does the modular design enable white-label solutions?

Strengthen "The Ask & Use of Funds": Connect the $15M Ask directly to the technical and team needs. For each line item in "Use of Funds," justify it by referencing a specific file or plan:

$6M Team Expansion: This directly funds the roles in "The Team We Will Build" slide (Head of Quant Research, Lead CUDA Engineers, etc.).

$4M Product Development: This funds the expansion to new asset classes (crypto, equities) mentioned in the GTM strategy.

$1.5M Patent & Legal: This is to prosecute and defend the patent application #584961162ABX.

IV. Founder & Team Story

Craft a Compelling Founder Narrative: The "Founder" slide is good but raw. Reframe it into a powerful story for investors. Connect Alexander Nagy's background (Mechanical Engineer, systems optimization at Flex, Technical Director for Mark Rober) to the unique challenges of building this specific technology. Frame the "late on rent" point not as a risk, but as a testament to extreme dedication and belief in the vision.

Target Audience & Tone

Audience: Sophisticated Series A investors. They are technically literate but time-poor. They need to see a clear vision backed by undeniable proof of execution.

Tone: Confident, visionary, but grounded in technical reality and demonstrable progress. Translate complex concepts into clear, high-impact statements.

Key Themes to Emphasize

The Problem: The "$7.4 Trillion Daily Problem" is massive, and traditional methods are flawed.

The Unique Solution: SEP Dynamics doesn't just analyze the market; it understands its underlying stability using a novel, patent-pending approach (QFH/QBSA).

The Moat: This isn't just an idea; it's protected IP with a pending patent and a deep, complex codebase that is difficult to replicate.

The Proof: The technology is live and has achieved 60.73% accuracy, a concrete metric that proves its efficacy.

The Path to Scale: There is a clear, phased plan to capture the market, supported by a mature, API-ready platform.

Final Deliverables

Based on your "clean sweep," please produce the following:

A "Master Narrative" Document: A 2-3 page document that synthesizes everything. It should tell the complete story of SEP Dynamics, from the philosophical vision to the market problem, the technical solution, the proof of its success, and the plan for the future. This will be our internal source of truth.

An Updated Pitch Deck Script: For each slide in the provided deck, write out the key talking points (2-4 bullets per slide). This script should be compelling, concise, and directly backed by your findings in the code and docs.

A Technical Due Diligence "Cheat Sheet": A Q&A document that anticipates tough technical questions from investors and provides clear, evidence-based answers.

Example Question: "You claim you have a GPU-accelerated engine. Where is the proof in your codebase?"

Example Answer: "The core CUDA kernels are located in src/apps/oanda_trader/. Specifically, tick_cuda_kernels.cu handles real-time data processing, and forward_window_kernels.cu performs our proprietary pattern analysis. The build system enables CUDA in src/apps/oanda_trader/CMakeLists.txt."