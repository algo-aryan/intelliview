// InterviewAI Pro - JavaScript Application Logic

class InterviewApp {
    constructor() {
        this.currentPage = 'landing';
        this.currentInterview = null;
        this.interviewTimer = null;
        this.questionTimer = null;
        this.currentQuestion = 1;
        this.totalQuestions = 8;
        this.interviewDuration = 0;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.showPage('landing');
        this.initializeComponents();
    }

    bindEvents() {
        // Navigation events
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.target.dataset.page;
                if (page) {
                    this.showPage(page);
                    this.updateNavigation(page);
                }
            });
        });

        // Action button events
        document.querySelectorAll('[data-action]').forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                this.handleAction(action, e.target);
            });
        });

        // Form events
        this.bindFormEvents();
        
        // Tab events
        this.bindTabEvents();
        
        // Interview card events
        this.bindInterviewCardEvents();
        
        // Upload area events
        this.bindUploadEvents();
        
        // Mobile menu toggle
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        if (mobileToggle) {
            mobileToggle.addEventListener('click', this.toggleMobileMenu.bind(this));
        }
    }

    bindFormEvents() {
        // Setup form validation and interactions
        const setupForm = document.querySelector('.setup-form');
        if (setupForm) {
            setupForm.addEventListener('input', this.validateSetupForm.bind(this));
            setupForm.addEventListener('change', this.updatePreview.bind(this));
        }

        // Search functionality
        const searchInput = document.querySelector('.search-box input');
        if (searchInput) {
            searchInput.addEventListener('input', this.handleSearch.bind(this));
        }

        // Filter functionality
        const filterSelect = document.querySelector('.history-actions select');
        if (filterSelect) {
            filterSelect.addEventListener('change', this.handleFilter.bind(this));
        }
    }

    bindTabEvents() {
        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    bindInterviewCardEvents() {
        document.querySelectorAll('.history-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (!e.target.closest('button')) {
                    const interviewId = card.dataset.interview;
                    this.showInterviewDetail(interviewId);
                }
            });
        });
    }

    bindUploadEvents() {
        const uploadArea = document.getElementById('resume-upload');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadArea.addEventListener('click', this.triggerFileSelect.bind(this));
        }

        const uploadBtn = document.querySelector('.upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.triggerFileSelect();
            });
        }
    }

    handleAction(action, element) {
        switch (action) {
            case 'start-interview':
                this.showPage('setup');
                this.updateNavigation('setup');
                break;
            case 'view-demo':
                this.showDemo();
                break;
            case 'start-interview-session':
                this.startInterview();
                break;
            case 'view-details':
                const card = element.closest('.history-card');
                const interviewId = card.dataset.interview;
                this.showInterviewDetail(interviewId);
                break;
            case 'back-to-list':
                this.showPage('history-list');
                this.updateNavigation('history-list');
                break;
            default:
                console.log('Action not implemented:', action);
        }
    }

    showPage(pageId) {
        // Hide all pages
        document.querySelectorAll('.page-section').forEach(page => {
            page.classList.remove('active');
        });

        // Show target page
        const targetPage = document.getElementById(`${pageId}-page`);
        if (targetPage) {
            targetPage.classList.add('active');
            this.currentPage = pageId;
        }
    }

    updateNavigation(activePageId) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.dataset.page === activePageId) {
                link.classList.add('active');
            }
        });
    }

    showDemo() {
        // Simulate demo modal or redirect
        alert('Demo functionality would open a video tutorial here.');
    }

    validateSetupForm() {
        const form = document.querySelector('.setup-form');
        const inputs = form.querySelectorAll('input, select');
        let isValid = true;

        inputs.forEach(input => {
            if (input.required && !input.value.trim()) {
                isValid = false;
            }
        });

        const startButton = document.querySelector('[data-action="start-interview-session"]');
        if (startButton) {
            startButton.disabled = !isValid;
            startButton.classList.toggle('btn--disabled', !isValid);
        }

        return isValid;
    }

    updatePreview() {
        // Update preview card based on form inputs
        const jobTitle = document.querySelector('input[placeholder*="Job Title"]')?.value;
        const interviewType = document.querySelector('select')?.value;
        
        // This would update preview content dynamically
        console.log('Updating preview for:', jobTitle, interviewType);
    }

    startInterview() {
        if (!this.validateSetupForm()) {
            alert('Please fill in all required fields before starting the interview.');
            return;
        }

        this.showPage('interview');
        this.initializeInterview();
    }

    initializeInterview() {
        this.currentQuestion = 1;
        this.interviewDuration = 0;
        this.updateInterviewProgress();
        this.startInterviewTimer();
        this.startQuestionTimer();
        this.animateMetrics();
    }

    updateInterviewProgress() {
        const progressFill = document.querySelector('.interview-header .progress-fill');
        const progressText = document.querySelector('.interview-header span');
        
        if (progressFill && progressText) {
            const percentage = (this.currentQuestion / this.totalQuestions) * 100;
            progressFill.style.width = `${percentage}%`;
            progressText.textContent = `Question ${this.currentQuestion} of ${this.totalQuestions}`;
        }
    }

    startInterviewTimer() {
        this.interviewTimer = setInterval(() => {
            this.interviewDuration++;
            this.updateTimerDisplay();
        }, 1000);
    }

    startQuestionTimer() {
        let questionTime = 0;
        this.questionTimer = setInterval(() => {
            questionTime++;
            // Could update a question-specific timer here
        }, 1000);
    }

    updateTimerDisplay() {
        const timer = document.querySelector('.interview-timer .timer');
        if (timer) {
            const minutes = Math.floor(this.interviewDuration / 60);
            const seconds = this.interviewDuration % 60;
            timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    animateMetrics() {
        // Simulate real-time metric updates
        const metrics = document.querySelectorAll('.metric-fill');
        
        setInterval(() => {
            metrics.forEach(metric => {
                const currentWidth = parseInt(metric.style.width) || 0;
                const variation = (Math.random() - 0.5) * 10; // ±5% variation
                const newWidth = Math.max(0, Math.min(100, currentWidth + variation));
                metric.style.width = `${newWidth}%`;
            });
        }, 2000);
    }

    nextQuestion() {
        if (this.currentQuestion < this.totalQuestions) {
            this.currentQuestion++;
            this.updateInterviewProgress();
            this.updateQuestionContent();
        } else {
            this.completeInterview();
        }
    }

    updateQuestionContent() {
        // Simulate updating question content
        const questions = [
            "Tell me about yourself and your background.",
            "Describe a challenging project you've worked on.",
            "How do you handle conflicts in a team?",
            "What are your greatest strengths and weaknesses?",
            "Where do you see yourself in 5 years?",
            "Describe a time when you had to learn something new quickly.",
            "How do you prioritize tasks when everything seems urgent?",
            "Tell me about a time you failed and what you learned."
        ];

        const questionText = document.querySelector('.question-text');
        if (questionText && questions[this.currentQuestion - 1]) {
            questionText.textContent = `"${questions[this.currentQuestion - 1]}"`;
        }
    }

    completeInterview() {
        clearInterval(this.interviewTimer);
        clearInterval(this.questionTimer);
        
        // Simulate saving interview results
        this.saveInterviewResults();
        
        // Redirect to history
        setTimeout(() => {
            this.showPage('history-list');
            this.updateNavigation('history-list');
        }, 2000);
        
        alert('Interview completed! Redirecting to your history...');
    }

    saveInterviewResults() {
        // Simulate saving interview data
        const interviewData = {
            date: new Date().toISOString().split('T')[0],
            duration: this.interviewDuration,
            questions: this.totalQuestions,
            score: Math.floor(Math.random() * 20) + 80 // Random score 80-100
        };
        
        console.log('Saving interview results:', interviewData);
    }

    showInterviewDetail(interviewId) {
        this.showPage('history-detail');
        this.loadInterviewData(interviewId);
    }

    loadInterviewData(interviewId) {
        // Simulate loading specific interview data
        const interviewData = {
            1: { title: 'Software Engineer Interview', date: 'December 15, 2024', score: 85 },
            2: { title: 'Product Manager Interview', date: 'December 10, 2024', score: 78 },
            3: { title: 'Team Lead Interview', date: 'December 5, 2024', score: 92 }
        };

        const data = interviewData[interviewId];
        if (data) {
            const titleElement = document.querySelector('.detail-header h1');
            const scoreElement = document.querySelector('.overall-score');
            
            if (titleElement) {
                titleElement.textContent = `${data.title} - ${data.date}`;
            }
            
            if (scoreElement) {
                scoreElement.textContent = data.score;
                scoreElement.className = `overall-score ${this.getScoreClass(data.score)}`;
            }
        }
    }

    getScoreClass(score) {
        if (score >= 90) return 'excellent';
        if (score >= 80) return 'good';
        if (score >= 70) return 'average';
        return 'poor';
    }

    switchTab(tabName) {
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Add active class to selected tab and content
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        const activeContent = document.getElementById(`${tabName}-tab`);

        if (activeTab) activeTab.classList.add('active');
        if (activeContent) activeContent.classList.add('active');
    }

    handleSearch(event) {
        const searchTerm = event.target.value.toLowerCase();
        const historyCards = document.querySelectorAll('.history-card');

        historyCards.forEach(card => {
            const title = card.querySelector('h3').textContent.toLowerCase();
            const isVisible = title.includes(searchTerm);
            card.style.display = isVisible ? 'block' : 'none';
        });
    }

    handleFilter(event) {
        const filterValue = event.target.value;
        const historyCards = document.querySelectorAll('.history-card');

        historyCards.forEach(card => {
            if (filterValue === 'All Types') {
                card.style.display = 'block';
            } else {
                const type = card.querySelector('.detail-item .value').textContent;
                card.style.display = type === filterValue ? 'block' : 'none';
            }
        });
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('drag-over');
    }

    handleFileDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('drag-over');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileUpload(files[0]);
        }
    }

    triggerFileSelect() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.pdf,.doc,.docx';
        input.onchange = (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        };
        input.click();
    }

    handleFileUpload(file) {
        // Validate file type and size
        const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const maxSize = 5 * 1024 * 1024; // 5MB

        if (!allowedTypes.includes(file.type)) {
            alert('Please upload a PDF, DOC, or DOCX file.');
            return;
        }

        if (file.size > maxSize) {
            alert('File size must be less than 5MB.');
            return;
        }

        // Simulate file upload
        this.showUploadProgress(file);
    }

    showUploadProgress(file) {
        const uploadArea = document.getElementById('resume-upload');
        const originalContent = uploadArea.innerHTML;

        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">📄</div>
                <h4>Uploading ${file.name}...</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        `;

        // Simulate upload progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 30;
            const progressFill = uploadArea.querySelector('.progress-fill');
            if (progressFill) {
                progressFill.style.width = `${Math.min(progress, 100)}%`;
            }

            if (progress >= 100) {
                clearInterval(interval);
                uploadArea.innerHTML = `
                    <div class="upload-content">
                        <div class="upload-icon">✅</div>
                        <h4>Resume uploaded successfully!</h4>
                        <p>${file.name}</p>
                        <button class="btn btn--outline btn--sm">Replace file</button>
                    </div>
                `;
                
                // Update step progress
                this.updateStepProgress(2);
            }
        }, 100);
    }

    updateStepProgress(stepNumber) {
        const steps = document.querySelectorAll('.step');
        steps.forEach((step, index) => {
            if (index < stepNumber) {
                step.classList.add('active');
            }
        });
    }

    toggleMobileMenu() {
        const navMenu = document.querySelector('.navbar-nav');
        if (navMenu) {
            navMenu.classList.toggle('mobile-open');
        }
    }

    initializeComponents() {
        // Initialize any additional components
        this.initializeHeroAnimation();
        this.initializeScrollEffects();
    }

    initializeHeroAnimation() {
        // Animate hero meters on page load
        const heroMeters = document.querySelectorAll('.hero-card .meter-fill');
        setTimeout(() => {
            heroMeters.forEach(meter => {
                const targetWidth = meter.style.width;
                meter.style.width = '0%';
                setTimeout(() => {
                    meter.style.width = targetWidth;
                }, 100);
            });
        }, 500);
    }

    initializeScrollEffects() {
        // Add scroll-based animations for feature cards
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.transform = 'translateY(0)';
                    entry.target.style.opacity = '1';
                }
            });
        }, observerOptions);

        document.querySelectorAll('.feature-card').forEach(card => {
            card.style.transform = 'translateY(20px)';
            card.style.opacity = '0';
            card.style.transition = 'transform 0.6s ease, opacity 0.6s ease';
            observer.observe(card);
        });
    }
}

// Utility functions
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m ${remainingSeconds}s`;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

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

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interviewApp = new InterviewApp();
    
    // Add some additional interactive behaviors
    addButtonHoverEffects();
    addCardHoverEffects();
    addFormValidationEffects();
});

function addButtonHoverEffects() {
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-1px)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

function addCardHoverEffects() {
    document.querySelectorAll('.feature-card, .history-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

function addFormValidationEffects() {
    document.querySelectorAll('.form-control').forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
            if (this.value) {
                this.parentElement.classList.add('filled');
            } else {
                this.parentElement.classList.remove('filled');
            }
        });
        
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.classList.remove('error');
                this.classList.add('valid');
            } else {
                this.classList.remove('valid');
                this.classList.add('error');
            }
        });
    });
}

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InterviewApp;
}