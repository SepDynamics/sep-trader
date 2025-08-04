// SEP Dynamics Website Main JavaScript

// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Active navigation highlighting
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
    
    function highlightActiveNav() {
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                currentSection = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', highlightActiveNav);
    
    // Initialize animations on scroll
    initScrollAnimations();
    
    // Initialize live trading terminal animation
    initLiveTradingTerminal();
    
    // Initialize performance counter animations
    initCounterAnimations();
    
    // Initialize form handling
    initFormHandling();
});

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.tech-card, .result-card, .flow-step, .platform-features');
    animateElements.forEach(el => observer.observe(el));
}

// Live trading terminal animation
function initLiveTradingTerminal() {
    const terminalContent = document.querySelector('.terminal-content');
    
    if (!terminalContent) return;
    
    const tradingMessages = [
        '[QuantumSignal] 🚀 MULTI-TIMEFRAME CONFIRMED: EUR_USD BUY',
        '[QuantumTracker] ✅ Trade executed: +$2,847 profit',
        '[QFH] Entropy: 0.923, Coherence: 0.856, Stability: 0.342',
        '[QuantumSignal] 🔮 PATTERN COLLAPSE PREDICTED: GBP_USD',
        '[QuantumTracker] ✅ Position closed: +$1,923 profit',
        '[System] Processing 47,234 patterns per minute',
        '[QuantumSignal] 🚀 HIGH CONFIDENCE: USD_JPY SELL',
        '[QuantumTracker] ✅ Trade executed: +$1,567 profit',
        '[QFH] Field coherence: 0.789, Rupture probability: 0.023',
        '[System] Daily P&L: +$7,234 | Win Rate: 73.2%'
    ];
    
    let messageIndex = 0;
    const lines = terminalContent.querySelectorAll('.terminal-line');
    
    function updateTerminalLine() {
        if (lines.length > 0) {
            const randomLine = lines[Math.floor(Math.random() * lines.length)];
            randomLine.textContent = tradingMessages[messageIndex];
            randomLine.style.color = getMessageColor(tradingMessages[messageIndex]);
            
            messageIndex = (messageIndex + 1) % tradingMessages.length;
        }
    }
    
    function getMessageColor(message) {
        if (message.includes('🚀') || message.includes('✅')) return '#00ff00';
        if (message.includes('🔮')) return '#ff6b6b';
        if (message.includes('QFH') || message.includes('System')) return '#4ecdc4';
        return '#00ff00';
    }
    
    // Update terminal every 3 seconds
    setInterval(updateTerminalLine, 3000);
}

// Counter animations
function initCounterAnimations() {
    const counters = document.querySelectorAll('.stat-number, .result-number, .metric-number');
    
    const observerOptions = {
        threshold: 0.5
    };
    
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                animateCounter(entry.target);
                entry.target.classList.add('counted');
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => counterObserver.observe(counter));
}

function animateCounter(element) {
    const text = element.textContent;
    const hasPercent = text.includes('%');
    const hasDollar = text.includes('$');
    const hasK = text.includes('K');
    const hasMs = text.includes('ms');
    
    // Extract number
    let number = parseFloat(text.replace(/[^0-9.]/g, ''));
    if (isNaN(number)) return;
    
    const duration = 2000; // 2 seconds
    const steps = 60;
    const increment = number / steps;
    let current = 0;
    
    const timer = setInterval(() => {
        current += increment;
        
        let displayValue = Math.min(current, number).toFixed(hasPercent || hasMs ? 2 : 0);
        
        // Add back formatting
        if (hasDollar && hasK) {
            displayValue = '$' + displayValue + 'K+';
        } else if (hasDollar) {
            displayValue = '$' + displayValue;
        } else if (hasPercent) {
            displayValue = displayValue + '%';
        } else if (hasMs) {
            displayValue = '<' + displayValue + 'ms';
        } else if (hasK) {
            displayValue = displayValue + '+';
        }
        
        element.textContent = displayValue;
        
        if (current >= number) {
            clearInterval(timer);
            // Restore original text to handle special formatting
            if (text.includes('24/7')) element.textContent = '24/7';
            if (text.includes('$7.4T')) element.textContent = '$7.4T';
        }
    }, duration / steps);
}

// Form handling
function initFormHandling() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            
            // Show loading state
            submitButton.textContent = 'Sending...';
            submitButton.disabled = true;
            
            // Simulate form submission
            setTimeout(() => {
                submitButton.textContent = 'Message Sent!';
                submitButton.style.backgroundColor = '#2ecc71';
                
                // Reset form
                setTimeout(() => {
                    form.reset();
                    submitButton.textContent = originalText;
                    submitButton.disabled = false;
                    submitButton.style.backgroundColor = '';
                }, 2000);
            }, 1500);
        });
    });
}

// Performance metrics updating
function initLiveMetrics() {
    const metrics = {
        accuracy: { element: document.querySelector('[data-metric="accuracy"]'), base: 60.73, variance: 0.5 },
        profitability: { element: document.querySelector('[data-metric="profitability"]'), base: 204.94, variance: 5 },
        signals: { element: document.querySelector('[data-metric="signals"]'), base: 19.1, variance: 1 }
    };
    
    function updateMetric(metric) {
        if (!metric.element) return;
        
        const variation = (Math.random() - 0.5) * metric.variance;
        const newValue = metric.base + variation;
        
        if (metric.element.textContent.includes('%')) {
            metric.element.textContent = newValue.toFixed(2) + '%';
        } else {
            metric.element.textContent = newValue.toFixed(2);
        }
    }
    
    // Update metrics every 10 seconds
    setInterval(() => {
        Object.values(metrics).forEach(updateMetric);
    }, 10000);
}

// Initialize live metrics if elements exist
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('[data-metric]')) {
        initLiveMetrics();
    }
});

// Parallax scrolling effect
function initParallaxScrolling() {
    const parallaxElements = document.querySelectorAll('.parallax');
    
    function updateParallax() {
        const scrolled = window.pageYOffset;
        
        parallaxElements.forEach(element => {
            const rate = scrolled * -0.5;
            element.style.transform = `translateY(${rate}px)`;
        });
    }
    
    window.addEventListener('scroll', updateParallax);
}

// Typing animation for hero text
function initTypingAnimation() {
    const typingElement = document.querySelector('.typing-text');
    
    if (!typingElement) return;
    
    const phrases = [
        'Quantum-Inspired Financial Intelligence',
        'Patent-Pending QFH Technology',
        '60.73% Prediction Accuracy',
        'Real-Time Pattern Collapse Detection'
    ];
    
    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    function typeText() {
        const currentPhrase = phrases[phraseIndex];
        
        if (isDeleting) {
            typingElement.textContent = currentPhrase.substring(0, charIndex - 1);
            charIndex--;
        } else {
            typingElement.textContent = currentPhrase.substring(0, charIndex + 1);
            charIndex++;
        }
        
        let timeout = isDeleting ? 50 : 100;
        
        if (!isDeleting && charIndex === currentPhrase.length) {
            timeout = 2000; // Pause at end
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            phraseIndex = (phraseIndex + 1) % phrases.length;
            timeout = 500; // Pause before next phrase
        }
        
        setTimeout(typeText, timeout);
    }
    
    typeText();
}

// Initialize all features
document.addEventListener('DOMContentLoaded', () => {
    initParallaxScrolling();
    initTypingAnimation();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initScrollAnimations,
        initLiveTradingTerminal,
        initCounterAnimations,
        initFormHandling
    };
}
