/**
 * JavaScript for Mobile Phone Price Predictor
 * Connects to FastAPI backend with modern async/await
 */

class PhonePricePredictor {
    constructor() {
        this.apiBaseUrl = '';  // Same origin
        this.form = document.getElementById('predictionForm');
        this.resultsSection = document.getElementById('resultsSection');
        this.predictBtn = document.getElementById('predictBtn');
        this.apiStatus = document.getElementById('apiStatus');

        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiStatus();
        this.loadExampleData();
        this.loadTrendingPhones();
    }

    bindEvents() {
        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.makePrediction();
        });

        // Load example button
        document.getElementById('loadExample').addEventListener('click', () => {
            this.loadExampleData();
        });

        // Clear form button
        document.getElementById('clearForm').addEventListener('click', () => {
            this.clearForm();
        });

        // Real-time form validation
        this.form.addEventListener('input', (e) => {
            this.validateInput(e.target);
        });
    }

    async checkApiStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (data.status === 'healthy' && data.model_loaded) {
                statusDot.className = 'status-dot online';
                statusText.textContent = 'API Online - Model Ready';
                statusDot.style.background = '#48bb78';
            } else {
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'API Issues - Model Not Ready';
                statusDot.style.background = '#f56565';
            }
        } catch (error) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'API Offline';
            statusDot.style.background = '#f56565';
            console.error('API health check failed:', error);
        }
    }

    async loadExampleData() {
        try {
            const response = await fetch('/api/examples');
            const data = await response.json();

            if (data.examples && data.examples.length > 0) {
                // Load the first example (mid-range phone)
                const example = data.examples[1]; // Mid-range phone
                this.populateForm(example.specs);
                this.showNotification('Example phone specs loaded!', 'success');
            }
        } catch (error) {
            console.error('Failed to load example data:', error);
            // Fallback to hardcoded example
            this.populateForm({
                battery_power: 3000,
                blue: 1,
                clock_speed: 2.0,
                dual_sim: 1,
                fc: 8,
                four_g: 1,
                int_memory: 64,
                m_dep: 0.7,
                mobile_wt: 180,
                n_cores: 4,
                pc: 12,
                px_height: 1920,
                px_width: 1080,
                ram: 3000,
                sc_h: 6.0,
                sc_w: 3.0,
                talk_time: 18,
                three_g: 1,
                touch_screen: 1,
                wifi: 1
            });
        }
    }

    populateForm(specs) {
        Object.keys(specs).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = specs[key] === 1;
                } else {
                    element.value = specs[key];
                }
            }
        });
    }

    clearForm() {
        this.form.reset();
        this.resultsSection.style.display = 'none';
        this.showNotification('Form cleared!', 'info');
    }

    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);

        if (input.type === 'number' && (value < min || value > max)) {
            input.style.borderColor = '#f56565';
            input.style.backgroundColor = '#fed7d7';
        } else {
            input.style.borderColor = '#e2e8f0';
            input.style.backgroundColor = 'white';
        }
    }

    async makePrediction() {
        // Show loading state
        this.setLoadingState(true);

        try {
            // Collect form data
            const formData = this.collectFormData();

            // Validate data
            if (!this.validateFormData(formData)) {
                this.showNotification('Please check all input values are within valid ranges.', 'error');
                this.setLoadingState(false);
                return;
            }

            // Make API request for recommendations
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);
            this.showNotification('Prediction completed successfully!', 'success');

        } catch (error) {
            console.error('Prediction failed:', error);
            this.showNotification('Prediction failed. Please try again.', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    collectFormData() {
        const formData = {};
        const form = this.form;

        // Collect numeric inputs
        const numericFields = [
            'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
            'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
            'ram', 'sc_h', 'sc_w', 'talk_time'
        ];

        numericFields.forEach(field => {
            const element = form.querySelector(`#${field}`);
            if (element) {
                formData[field] = parseFloat(element.value) || 0;
            }
        });

        // Collect checkbox inputs
        const checkboxFields = [
            'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'
        ];

        checkboxFields.forEach(field => {
            const element = form.querySelector(`#${field}`);
            if (element) {
                formData[field] = element.checked ? 1 : 0;
            }
        });

        return formData;
    }

    validateFormData(data) {
        const validations = {
            battery_power: { min: 500, max: 5000 },
            clock_speed: { min: 0.5, max: 3.0 },
            fc: { min: 0, max: 50 },
            int_memory: { min: 1, max: 512 },
            m_dep: { min: 0.1, max: 2.0 },
            mobile_wt: { min: 80, max: 300 },
            n_cores: { min: 1, max: 8 },
            pc: { min: 0, max: 100 },
            px_height: { min: 100, max: 4000 },
            px_width: { min: 100, max: 4000 },
            ram: { min: 256, max: 16000 },
            sc_h: { min: 3.0, max: 8.0 },
            sc_w: { min: 2.0, max: 6.0 },
            talk_time: { min: 1, max: 30 }
        };

        for (const [field, rules] of Object.entries(validations)) {
            const value = data[field];
            if (value < rules.min || value > rules.max) {
                console.error(`${field} value ${value} is out of range [${rules.min}, ${rules.max}]`);
                return false;
            }
        }

        return true;
    }

    displayResults(result) {
        // Show results section
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Update price category and range
        document.getElementById('priceCategory').textContent = result.predicted_category;
        document.getElementById('priceRange').textContent = result.price_range;

        // Update confidence meter
        const confidence = Math.round(result.confidence * 100);
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');

        // Animate confidence bar
        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
        }, 100);

        confidenceValue.textContent = `${confidence}%`;

        // Color code confidence
        if (confidence >= 90) {
            confidenceFill.style.background = 'linear-gradient(45deg, #48bb78, #38a169)';
        } else if (confidence >= 70) {
            confidenceFill.style.background = 'linear-gradient(45deg, #ed8936, #dd6b20)';
        } else {
            confidenceFill.style.background = 'linear-gradient(45deg, #f56565, #e53e3e)';
        }

        // Update processing time
        document.getElementById('processingTime').textContent = `${result.processing_time_ms} ms`;

        // Update model accuracy
        document.getElementById('modelAccuracy').textContent = result.model_accuracy;

        // Display phone recommendations if available
        if (result.recommended_phones && result.recommended_phones.length > 0) {
            this.displayRecommendations(result.recommended_phones);
        }

        // Add result animation
        this.resultsSection.style.animation = 'none';
        setTimeout(() => {
            this.resultsSection.style.animation = 'slideIn 0.5s ease';
        }, 10);
    }

    setLoadingState(loading) {
        const btn = this.predictBtn;
        const spinner = btn.querySelector('.loading-spinner');
        const text = btn.querySelector('span');

        if (loading) {
            btn.classList.add('loading');
            btn.disabled = true;
            spinner.style.display = 'block';
            text.style.display = 'none';
        } else {
            btn.classList.remove('loading');
            btn.disabled = false;
            spinner.style.display = 'none';
            text.style.display = 'block';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add styles
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: this.getNotificationColor(type),
            color: 'white',
            padding: '1rem 1.5rem',
            borderRadius: '8px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            zIndex: '10000',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease',
            maxWidth: '300px'
        });

        // Add to DOM
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after delay
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }

    getNotificationColor(type) {
        switch (type) {
            case 'success': return '#48bb78';
            case 'error': return '#f56565';
            case 'warning': return '#ed8936';
            default: return '#4299e1';
        }
    }

    displayRecommendations(phones) {
        const recommendationsSection = document.getElementById('recommendationsSection');
        const recommendationsGrid = document.getElementById('recommendationsGrid');

        if (!phones || phones.length === 0) {
            recommendationsSection.style.display = 'none';
            return;
        }

        // Clear existing recommendations
        recommendationsGrid.innerHTML = '';

        // Create phone cards
        phones.forEach(phone => {
            const phoneCard = this.createPhoneCard(phone);
            recommendationsGrid.appendChild(phoneCard);
        });

        // Show recommendations section
        recommendationsSection.style.display = 'block';

        // Animate cards
        const cards = recommendationsGrid.querySelectorAll('.phone-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            setTimeout(() => {
                card.style.transition = 'all 0.3s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    createPhoneCard(phone) {
        const card = document.createElement('div');
        card.className = 'phone-card';

        const similarityBadge = this.getSimilarityBadge(phone.similarity_score);
        const ratingStars = this.generateStars(phone.rating);

        card.innerHTML = `
            <div class="phone-card-header">
                <div class="phone-image">
                    <i class="fas fa-mobile-alt"></i>
                </div>
                <div class="similarity-badge ${similarityBadge.class}">
                    ${phone.similarity_score}% Match
                </div>
            </div>
            <div class="phone-card-content">
                <div class="phone-brand">${phone.brand}</div>
                <div class="phone-model">${phone.model}</div>
                <div class="phone-price">${phone.formatted_price}</div>
                <div class="phone-rating">
                    <div class="stars">${ratingStars}</div>
                    <span class="rating-count">(${phone.reviews_count})</span>
                </div>
                <div class="phone-specs">
                    <div class="spec-item">
                        <i class="fas fa-memory"></i>
                        <span>${phone.specifications.RAM}</span>
                    </div>
                    <div class="spec-item">
                        <i class="fas fa-hdd"></i>
                        <span>${phone.specifications.Storage}</span>
                    </div>
                    <div class="spec-item">
                        <i class="fas fa-battery-full"></i>
                        <span>${phone.specifications.Battery}</span>
                    </div>
                    <div class="spec-item">
                        <i class="fas fa-camera"></i>
                        <span>${phone.specifications.Camera}</span>
                    </div>
                </div>
                <div class="phone-features">
                    ${phone.key_features.slice(0, 3).map(feature =>
            `<span class="feature-tag">${feature}</span>`
        ).join('')}
                </div>
                <div class="phone-availability ${phone.availability.toLowerCase().replace(' ', '-')}">
                    ${phone.availability}
                </div>
            </div>
        `;

        return card;
    }

    getSimilarityBadge(score) {
        if (score >= 90) return { class: 'excellent', text: 'Excellent Match' };
        if (score >= 80) return { class: 'good', text: 'Good Match' };
        if (score >= 70) return { class: 'fair', text: 'Fair Match' };
        return { class: 'basic', text: 'Basic Match' };
    }

    generateStars(rating) {
        const fullStars = Math.floor(rating);
        const hasHalfStar = rating % 1 >= 0.5;
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

        let stars = '';
        for (let i = 0; i < fullStars; i++) {
            stars += '<i class="fas fa-star"></i>';
        }
        if (hasHalfStar) {
            stars += '<i class="fas fa-star-half-alt"></i>';
        }
        for (let i = 0; i < emptyStars; i++) {
            stars += '<i class="far fa-star"></i>';
        }

        return stars;
    }

    async loadTrendingPhones() {
        const trendingGrid = document.getElementById('trendingGrid');
        const loadingSpinner = document.getElementById('trendingLoading');

        try {
            const response = await fetch('/api/phones/trending');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Clear existing content
            trendingGrid.innerHTML = '';
            loadingSpinner.style.display = 'none';

            // Create trending phone cards
            data.trending_phones.forEach(phone => {
                const phoneCard = this.createTrendingPhoneCard(phone);
                trendingGrid.appendChild(phoneCard);
            });

        } catch (error) {
            console.error('Failed to load trending phones:', error);
            loadingSpinner.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Failed to load trending phones</span>';
        }
    }

    createTrendingPhoneCard(phone) {
        const card = document.createElement('div');
        card.className = 'trending-phone-card';

        const ratingStars = this.generateStars(phone.rating);

        card.innerHTML = `
            <div class="trending-phone-content">
                <div class="phone-image">
                    <i class="fas fa-mobile-alt"></i>
                </div>
                <div class="phone-info">
                    <div class="phone-brand">${phone.brand}</div>
                    <div class="phone-model">${phone.model}</div>
                    <div class="phone-price">${phone.formatted_price}</div>
                    <div class="phone-rating">
                        <div class="stars">${ratingStars}</div>
                        <span class="rating-text">${phone.rating}</span>
                    </div>
                    <div class="phone-features">
                        ${phone.key_features.map(feature =>
            `<span class="feature-tag">${feature}</span>`
        ).join('')}
                    </div>
                </div>
            </div>
        `;

        return card;
    }
}

// Advanced features
class AdvancedFeatures {
    constructor(predictor) {
        this.predictor = predictor;
        this.init();
    }

    init() {
        this.addKeyboardShortcuts();
        this.addFormAutoSave();
        this.addPerformanceMonitoring();
    }

    addKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Enter to predict
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.predictor.makePrediction();
            }

            // Ctrl+R to reset form
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.predictor.clearForm();
            }

            // Ctrl+E to load example
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                this.predictor.loadExampleData();
            }
        });
    }

    addFormAutoSave() {
        const form = this.predictor.form;
        const saveKey = 'phonePredictorFormData';

        // Load saved data on page load
        const savedData = localStorage.getItem(saveKey);
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                this.predictor.populateForm(data);
            } catch (error) {
                console.warn('Failed to load saved form data:', error);
            }
        }

        // Save data on input
        form.addEventListener('input', debounce(() => {
            const formData = this.predictor.collectFormData();
            localStorage.setItem(saveKey, JSON.stringify(formData));
        }, 1000));
    }

    addPerformanceMonitoring() {
        // Monitor API response times
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const start = performance.now();
            const response = await originalFetch(...args);
            const end = performance.now();

            if (args[0].includes('/api/predict')) {
                console.log(`Prediction API call took ${Math.round(end - start)}ms`);
            }

            return response;
        };
    }
}

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

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    const predictor = new PhonePricePredictor();
    const advanced = new AdvancedFeatures(predictor);

    console.log('🚀 Mobile Phone Price Predictor loaded successfully!');
    console.log('📱 Keyboard shortcuts:');
    console.log('   Ctrl+Enter: Make prediction');
    console.log('   Ctrl+R: Reset form');
    console.log('   Ctrl+E: Load example');
});

// Service Worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
