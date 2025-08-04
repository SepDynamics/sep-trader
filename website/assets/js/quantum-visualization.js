// Quantum Field Visualization for SEP Dynamics Website
// Creates animated quantum field effect using Three.js

class QuantumFieldVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particles = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.container.clientWidth / this.container.clientHeight, 
            0.1, 
            1000
        );
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            alpha: true, 
            antialias: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);
        
        // Create quantum field particles
        this.createQuantumField();
        
        // Position camera
        this.camera.position.z = 5;
        
        // Start animation
        this.animate();
        
        // Handle resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    createQuantumField() {
        const particleCount = 2000;
        const geometry = new THREE.BufferGeometry();
        
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        // Create particle positions and properties
        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            
            // Random positions in space
            positions[i3] = (Math.random() - 0.5) * 10;
            positions[i3 + 1] = (Math.random() - 0.5) * 10;
            positions[i3 + 2] = (Math.random() - 0.5) * 10;
            
            // Quantum-inspired colors (blue to purple gradient)
            const colorIntensity = Math.random();
            colors[i3] = 0.3 + colorIntensity * 0.4;     // R
            colors[i3 + 1] = 0.5 + colorIntensity * 0.3; // G  
            colors[i3 + 2] = 0.8 + colorIntensity * 0.2; // B
            
            // Random sizes
            sizes[i] = Math.random() * 2 + 1;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Particle material with quantum glow effect
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                pixelRatio: { value: window.devicePixelRatio }
            },
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                
                varying vec3 vColor;
                varying float vAlpha;
                
                uniform float time;
                
                void main() {
                    vColor = color;
                    
                    vec3 pos = position;
                    
                    // Quantum field harmonics effect
                    pos.x += sin(time * 0.5 + position.y * 0.5) * 0.3;
                    pos.y += cos(time * 0.3 + position.x * 0.7) * 0.2;
                    pos.z += sin(time * 0.4 + position.x * 0.3) * 0.25;
                    
                    // Pulsating effect for quantum coherence
                    float pulse = sin(time * 2.0 + length(position) * 0.5) * 0.3 + 0.7;
                    vAlpha = pulse;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    gl_PointSize = size * pulse * (300.0 / -mvPosition.z);
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vAlpha;
                
                void main() {
                    float distanceToCenter = length(gl_PointCoord - vec2(0.5));
                    
                    if (distanceToCenter > 0.5) discard;
                    
                    // Quantum glow effect
                    float glow = 1.0 - smoothstep(0.0, 0.5, distanceToCenter);
                    float intensity = pow(glow, 2.0) * vAlpha;
                    
                    gl_FragColor = vec4(vColor * intensity, intensity * 0.8);
                }
            `,
            transparent: true,
            vertexColors: true,
            blending: THREE.AdditiveBlending
        });
        
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
        
        // Add quantum field lines
        this.createFieldLines();
    }
    
    createFieldLines() {
        const lineCount = 50;
        
        for (let i = 0; i < lineCount; i++) {
            const geometry = new THREE.BufferGeometry();
            const pointCount = 20;
            const positions = new Float32Array(pointCount * 3);
            
            // Create curved field lines
            for (let j = 0; j < pointCount; j++) {
                const t = j / (pointCount - 1);
                const angle = i * 0.3 + Math.PI * 2 * t;
                const radius = 2 + Math.sin(t * Math.PI * 2) * 0.5;
                
                positions[j * 3] = Math.cos(angle) * radius;
                positions[j * 3 + 1] = (t - 0.5) * 8;
                positions[j * 3 + 2] = Math.sin(angle) * radius;
            }
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const material = new THREE.LineBasicMaterial({
                color: new THREE.Color(0.2, 0.4, 0.8),
                transparent: true,
                opacity: 0.1
            });
            
            const line = new THREE.Line(geometry, material);
            this.scene.add(line);
        }
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const time = Date.now() * 0.001;
        
        // Update particle system
        if (this.particles && this.particles.material.uniforms) {
            this.particles.material.uniforms.time.value = time;
        }
        
        // Rotate the entire field slowly
        if (this.particles) {
            this.particles.rotation.y = time * 0.1;
            this.particles.rotation.x = Math.sin(time * 0.05) * 0.2;
        }
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// Simple fallback for browsers without WebGL support
class FallbackVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.createFallback();
    }
    
    createFallback() {
        // Create CSS-based animation fallback
        const canvas = document.createElement('canvas');
        canvas.width = this.container.clientWidth;
        canvas.height = this.container.clientHeight;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        const ctx = canvas.getContext('2d');
        this.container.appendChild(canvas);
        
        const particles = [];
        for (let i = 0; i < 100; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                size: Math.random() * 3 + 1,
                opacity: Math.random() * 0.5 + 0.3
            });
        }
        
        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                
                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(74, 144, 226, ${particle.opacity})`;
                ctx.fill();
            });
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
}

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const quantumFieldContainer = document.getElementById('quantumField');
    
    if (quantumFieldContainer) {
        // Check for WebGL support
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (gl && typeof THREE !== 'undefined') {
            // Use Three.js visualization
            new QuantumFieldVisualization('quantumField');
        } else {
            // Use fallback visualization
            new FallbackVisualization('quantumField');
        }
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumFieldVisualization, FallbackVisualization };
}
