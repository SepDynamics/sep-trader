// SEP Dynamics Website JavaScript - Enhanced Version

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initQuantumVisualization();
    initScrollAnimations();
    initMetricsAnimation();
});

// Navigation functionality
function initNavigation() {
    // Smooth scroll for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Active navigation highlighting
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-menu a[href^="#"]');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.pageYOffset >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${current}`) {
                item.classList.add('active');
            }
        });
    });
}

// Quantum field visualization
function initQuantumVisualization() {
    const canvas = document.getElementById('quantumField');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match container
    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Animation state
    let time = 0;
    const particles = [];
    const connections = [];
    
    // Initialize particles
    for (let i = 0; i < 50; i++) {
        particles.push({
            x: Math.random() * canvas.width / window.devicePixelRatio,
            y: Math.random() * canvas.height / window.devicePixelRatio,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 3 + 1,
            coherence: Math.random(),
            stability: Math.random(),
            entropy: Math.random(),
            phase: Math.random() * Math.PI * 2
        });
    }

    function animate() {
        time += 0.016; // ~60fps
        const width = canvas.width / window.devicePixelRatio;
        const height = canvas.height / window.devicePixelRatio;
        
        // Clear canvas with fade effect
        ctx.fillStyle = 'rgba(10, 10, 10, 0.1)';
        ctx.fillRect(0, 0, width, height);
        
        // Draw quantum field grid
        ctx.strokeStyle = 'rgba(102, 126, 234, 0.1)';
        ctx.lineWidth = 0.5;
        const gridSize = 40;
        
        for (let x = 0; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        for (let y = 0; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Update and draw particles
        particles.forEach((particle, index) => {
            // Update quantum properties
            particle.phase += 0.02;
            particle.coherence = 0.5 + 0.3 * Math.sin(time + particle.phase);
            particle.stability = 0.5 + 0.2 * Math.cos(time * 0.7 + particle.phase);
            particle.entropy = 0.3 + 0.2 * Math.sin(time * 1.3 + particle.phase);
            
            // Update position with quantum field influence
            const fieldStrength = particle.coherence * particle.stability;
            particle.vx += (Math.random() - 0.5) * 0.01 * (1 - fieldStrength);
            particle.vy += (Math.random() - 0.5) * 0.01 * (1 - fieldStrength);
            
            // Apply field harmonics
            particle.x += particle.vx + 0.2 * Math.sin(time + particle.x * 0.01);
            particle.y += particle.vy + 0.2 * Math.cos(time + particle.y * 0.01);
            
            // Boundary conditions
            if (particle.x < 0 || particle.x > width) particle.vx *= -0.8;
            if (particle.y < 0 || particle.y > height) particle.vy *= -0.8;
            particle.x = Math.max(0, Math.min(width, particle.x));
            particle.y = Math.max(0, Math.min(height, particle.y));
            
            // Draw particle with quantum state visualization
            const alpha = particle.coherence * 0.8;
            const radius = particle.size * (0.5 + particle.stability * 0.5);
            
            // Particle core
            const gradient = ctx.createRadialGradient(
                particle.x, particle.y, 0,
                particle.x, particle.y, radius * 3
            );
            gradient.addColorStop(0, `rgba(102, 126, 234, ${alpha})`);
            gradient.addColorStop(0.5, `rgba(118, 75, 162, ${alpha * 0.6})`);
            gradient.addColorStop(1, `rgba(102, 126, 234, 0)`);
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, radius * 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Bright center
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.8})`;
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
            ctx.fill();
            
            // Quantum entanglement visualization
            particles.forEach((otherParticle, otherIndex) => {
                if (index >= otherIndex) return;
                
                const dx = particle.x - otherParticle.x;
                const dy = particle.y - otherParticle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 120) {
                    const entanglement = (particle.coherence + otherParticle.coherence) / 2;
                    const connectionAlpha = (1 - distance / 120) * entanglement * 0.3;
                    
                    if (connectionAlpha > 0.05) {
                        ctx.strokeStyle = `rgba(102, 126, 234, ${connectionAlpha})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particle.x, particle.y);
                        ctx.lineTo(otherParticle.x, otherParticle.y);
                        ctx.stroke();
                        
                        // Quantum flux visualization
                        const midX = (particle.x + otherParticle.x) / 2;
                        const midY = (particle.y + otherParticle.y) / 2;
                        const flux = Math.sin(time * 3 + distance * 0.1) * 0.5 + 0.5;
                        
                        ctx.fillStyle = `rgba(255, 255, 255, ${connectionAlpha * flux})`;
                        ctx.beginPath();
                        ctx.arc(midX, midY, 1, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });
        });
        
        // Calculate and display system metrics
        updateQuantumMetrics(particles);
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function updateQuantumMetrics(particles) {
    if (!particles || particles.length === 0) return;
    
    // Calculate system-wide quantum metrics
    const totalParticles = particles.length;
    const avgCoherence = particles.reduce((sum, p) => sum + p.coherence, 0) / totalParticles;
    const avgStability = particles.reduce((sum, p) => sum + p.stability, 0) / totalParticles;
    const avgEntropy = particles.reduce((sum, p) => sum + p.entropy, 0) / totalParticles;
    
    // Update any displayed metrics if elements exist
    const coherenceEl = document.getElementById('coherence-value');
    const stabilityEl = document.getElementById('stability-value');
    const entropyEl = document.getElementById('entropy-value');
    
    if (coherenceEl) coherenceEl.textContent = avgCoherence.toFixed(3);
    if (stabilityEl) stabilityEl.textContent = avgStability.toFixed(3);
    if (entropyEl) entropyEl.textContent = avgEntropy.toFixed(3);
}

// Animated metrics counters
function initMetricsAnimation() {
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateMetric(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe stat numbers and result numbers
    const metrics = document.querySelectorAll('.stat-number, .result-number, .metric-number');
    metrics.forEach(metric => observer.observe(metric));
}

function animateMetric(element) {
    const target = element.textContent;
    const isPercentage = target.includes('%');
    const isTime = target.includes('ms') || target.includes('s');
    const isMoney = target.includes('$') || target.includes('K') || target.includes('M') || target.includes('T');
    
    let numericValue = parseFloat(target.replace(/[^0-9.]/g, ''));
    if (isNaN(numericValue)) return;
    
    let current = 0;
    const increment = numericValue / 60; // 1 second animation at 60fps
    
    function updateCounter() {
        current += increment;
        if (current >= numericValue) {
            current = numericValue;
        }
        
        let displayValue = current.toFixed(isPercentage ? 2 : 0);
        
        if (isMoney) {
            if (target.includes('T')) {
                displayValue = '$' + displayValue + 'T';
            } else if (target.includes('M')) {
                displayValue = '$' + displayValue + 'M';
            } else if (target.includes('K')) {
                displayValue = '$' + displayValue + 'K+';
            } else {
                displayValue = '$' + displayValue;
            }
        } else if (isPercentage) {
            displayValue += '%';
        } else if (isTime) {
            if (target.includes('ms')) {
                displayValue = '<' + Math.ceil(current) + 'ms';
            } else {
                displayValue += 's';
            }
        } else if (target.includes('+')) {
            displayValue += '+';
        }
        
        element.textContent = displayValue;
        
        if (current < numericValue) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    updateCounter();
}

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                
                // Special animations for specific elements
                if (entry.target.classList.contains('tech-card')) {
                    // Stagger animation for tech cards
                    const index = Array.from(entry.target.parentNode.children).indexOf(entry.target);
                    entry.target.style.animationDelay = `${index * 0.1}s`;
                }
                
                if (entry.target.classList.contains('result-card')) {
                    // Stagger animation for result cards
                    const index = Array.from(entry.target.parentNode.children).indexOf(entry.target);
                    entry.target.style.animationDelay = `${index * 0.15}s`;
                }
            }
        });
    }, observerOptions);
    
    // Observe all animatable elements
    const animatedElements = document.querySelectorAll(`
        .tech-card, 
        .result-card, 
        .feature,
        .problem-column,
        .solution-column,
        .opportunity-card,
        .resource-card,
        .timeline-item,
        .platform-features,
        .hero-stats .stat,
        .hero-metrics .metric
    `);
    
    animatedElements.forEach(el => observer.observe(el));
}

// Terminal simulation for dashboard preview
function initTerminalSimulation() {
    const terminalContent = document.querySelector('.terminal-content');
    if (!terminalContent) return;
    
    const messages = [
        { text: '[QuantumSignal] 🚀 MULTI-TIMEFRAME CONFIRMED: EUR_USD BUY', type: 'success' },
        { text: '[QuantumTracker] ✅ Trade executed: +$2,847 profit', type: 'profit' },
        { text: '[QFH] Pattern collapse detected: GBP_USD SELL signal', type: 'warning' },
        { text: '[QuantumTracker] ✅ Position closed: +$1,923 profit', type: 'profit' },
        { text: '[QFH] Entropy: 0.923, Coherence: 0.856, Stability: 0.342', type: 'info' },
        { text: '[System] Daily P&L: +$7,234 | Win Rate: 73.2%', type: 'success' },
        { text: '[RemoteData] Cloud sync complete | 16 pairs active', type: 'info' },
        { text: '[System] Status: PROFITABLE | Next signal in 47 seconds', type: 'info' }
    ];
    
    let currentIndex = 0;
    
    function addMessage() {
        if (currentIndex < messages.length) {
            const message = messages[currentIndex];
            const line = document.createElement('div');
            line.className = 'terminal-line';
            line.textContent = message.text;
            
            // Add color based on message type
            switch (message.type) {
                case 'success':
                    line.style.color = '#22c55e';
                    break;
                case 'profit':
                    line.style.color = '#667eea';
                    break;
                case 'warning':
                    line.style.color = '#fbbf24';
                    break;
                case 'info':
                default:
                    line.style.color = '#ccc';
            }
            
            terminalContent.appendChild(line);
            currentIndex++;
            
            // Auto-scroll to bottom
            terminalContent.scrollTop = terminalContent.scrollHeight;
            
            // Schedule next message
            setTimeout(addMessage, 1500 + Math.random() * 1000);
        } else {
            // Reset and start over
            setTimeout(() => {
                terminalContent.innerHTML = '';
                currentIndex = 0;
                addMessage();
            }, 3000);
        }
    }
    
    // Start the simulation when the terminal comes into view
    const terminalObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setTimeout(addMessage, 1000);
                terminalObserver.unobserve(entry.target);
            }
        });
    });
    
    terminalObserver.observe(terminalContent);
}

// Enhanced form handling
function initContactForm() {
    const form = document.querySelector('.contact-form form');
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        
        // Simple validation
        const required = ['name', 'email', 'company', 'interest'];
        const missing = required.filter(field => !data[field]);
        
        if (missing.length > 0) {
            alert(`Please fill in the following fields: ${missing.join(', ')}`);
            return;
        }
        
        // Simulate form submission
        const button = form.querySelector('button[type="submit"]');
        const originalText = button.textContent;
        button.textContent = 'Sending...';
        button.disabled = true;
        
        setTimeout(() => {
            alert('Thank you for your interest! We will contact you within 24 hours.');
            form.reset();
            button.textContent = originalText;
            button.disabled = false;
        }, 2000);
    });
}

// Initialize additional features on page load
document.addEventListener('DOMContentLoaded', function() {
    initTerminalSimulation();
    initContactForm();
    
    // Add hover effects to cards
    const cards = document.querySelectorAll('.tech-card, .result-card, .opportunity-card, .resource-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize
window.addEventListener('resize', debounce(() => {
    // Reinitialize components that depend on window size
    const canvas = document.getElementById('quantumField');
    if (canvas) {
        // Canvas will be resized by the resize handler in initQuantumVisualization
    }
}, 250));

// Performance optimization
if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
        // Initialize non-critical features when browser is idle
        console.log('SEP Dynamics website loaded successfully');
    });
}
