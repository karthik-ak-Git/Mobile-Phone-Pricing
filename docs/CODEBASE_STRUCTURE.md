# 🗂️ Codebase Structure Documentation

## Overview

This document provides a comprehensive guide to the Mobile Phone Pricing project's codebase structure, explaining the purpose of each directory and file.

## Directory Structure

### 📁 Root Directory
```
Mobile-Phone-Pricing/
├── README.md              # Main project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── start_server.bat      # Windows server startup script
└── start_server.sh       # Unix/Mac server startup script
```

### 📁 api/
**Purpose**: API implementation and endpoints

```
api/
└── main_api.py           # FastAPI application with all endpoints
```

**Key Components:**
- FastAPI application setup
- Health check endpoint (`/health`)
- Prediction endpoint (`/api/recommend`)
- Example data endpoint (`/api/examples`)
- Trending phones endpoint (`/api/phones/trending`)
- CORS middleware configuration
- Static file serving

### 📁 data/
**Purpose**: Data loading and preprocessing utilities

```
data/
└── dataloader.py         # Dataset loading and preprocessing
```

**Key Components:**
- CSV data loading
- Feature normalization
- Train/test splitting
- Data validation

### 📁 dataset/
**Purpose**: Training and testing datasets

```
dataset/
├── train.csv             # Training dataset (2000 samples)
└── test.csv              # Testing dataset (500 samples)
```

**Dataset Features:**
- 20 input features
- 4 price categories (0-3)
- Balanced distribution
- Real-world phone specifications

### 📁 docs/
**Purpose**: Project documentation and summaries

```
docs/
├── README.md                      # Original documentation
├── README_NEW.md                  # Updated documentation
├── CLEANUP_SUMMARY.md             # Code cleanup notes
├── CLEAN_CODEBASE_SUMMARY.md      # Clean code summary
├── ENHANCED_SYSTEM_SUMMARY.md     # System enhancements
├── PROJECT_COMPLETION_SUMMARY.md  # Project completion notes
├── CODEBASE_STRUCTURE.md          # This file
└── Predict Mobile Phone Pricing.pdf  # Project presentation
```

**Documentation Types:**
- Technical specifications
- Implementation details
- Enhancement summaries
- Project milestones

### 📁 frontend/
**Purpose**: Web interface (HTML, CSS, JavaScript)

```
frontend/
├── index.html            # Main HTML (10/10 accessibility)
├── styles.css            # Enhanced CSS with responsive design
└── scripts.js            # JavaScript with validation
```

#### index.html Features:
- WCAG 2.1 AA compliant
- Semantic HTML5 structure
- ARIA labels and roles
- Skip navigation links
- Meta tags for SEO
- Responsive viewport

#### styles.css Features:
- Modern gradient design
- Flexbox and Grid layouts
- CSS animations
- Responsive breakpoints (480px, 768px)
- Dark mode support
- Print styles
- Accessibility focus styles

#### scripts.js Features:
- ES6+ JavaScript
- Async/await API calls
- Real-time validation
- Form auto-save
- Keyboard shortcuts
- Screen reader support
- Error handling
- Notification system

### 📁 models/
**Purpose**: Trained PyTorch models

```
models/
├── optimized_model.pth       # Best model (99% accuracy)
├── enhanced_model.pth        # Enhanced version
├── advanced_dnn_model.pth    # Advanced architecture
└── simple_dnn_model.pth      # Baseline model
```

**Model Architecture:**
```
Input (20 features) → 
Dense(128) + ReLU + Dropout(0.3) →
Dense(64) + ReLU + Dropout(0.2) →
Dense(32) + ReLU →
Output(4 classes)
```

### 📁 outputs/
**Purpose**: Generated outputs and visualizations

```
outputs/
└── plots/                    # Training/evaluation plots
    ├── confusion_matrix.png
    ├── training_history.png
    └── feature_importance.png
```

### 📁 scripts/
**Purpose**: Utility scripts for testing, evaluation, and optimization

```
scripts/
├── test_recommendations.py   # Test recommendation system
├── test_system.py           # System integration tests
├── demo_final_model.py      # Model demonstration
├── evaluate.py              # Model evaluation
├── evaluate_enhanced.py     # Enhanced evaluation
├── enhance_model.py         # Model enhancement
├── optimize_model.py        # Model optimization
└── final_evaluation.py      # Final evaluation metrics
```

**Script Purposes:**

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_recommendations.py` | Test recommendation algorithm | After model changes |
| `test_system.py` | End-to-end system testing | Before deployment |
| `demo_final_model.py` | Interactive model demo | Demonstrations |
| `evaluate.py` | Basic model evaluation | Model validation |
| `evaluate_enhanced.py` | Detailed evaluation | Performance analysis |
| `enhance_model.py` | Model improvement | Model development |
| `optimize_model.py` | Model optimization | Performance tuning |
| `final_evaluation.py` | Complete evaluation suite | Final testing |

### 📄 Root Python Files

```
Root/
├── app.py                      # Flask application (alternative)
├── api.py                      # Alternative API implementation
├── main.py                     # Main entry point
├── mobile_phone_predictor.py   # Core prediction logic
└── phone_database.py           # Phone database management
```

## File Dependencies

### Dependency Graph

```
index.html
    ├── styles.css
    └── scripts.js
        └── main_api.py
            ├── mobile_phone_predictor.py
            │   └── optimized_model.pth
            └── phone_database.py
                └── dataset/train.csv
```

## Code Organization Principles

### 1. Separation of Concerns
- **Frontend**: Pure presentation layer (HTML/CSS/JS)
- **API**: Business logic and routing (FastAPI)
- **Models**: ML models and weights (PyTorch)
- **Data**: Dataset and preprocessing
- **Scripts**: Utilities and testing

### 2. Modularity
- Each module has a single responsibility
- Minimal coupling between modules
- Easy to test and maintain

### 3. Accessibility First
- WCAG 2.1 AA compliance
- Semantic HTML
- ARIA attributes
- Keyboard navigation

### 4. Performance
- Optimized models
- Lazy loading
- Caching strategies
- Minified assets

## Development Workflow

### 1. Data Changes
```
dataset/ → data/dataloader.py → scripts/evaluate.py
```

### 2. Model Changes
```
scripts/enhance_model.py → models/ → scripts/evaluate_enhanced.py
```

### 3. API Changes
```
api/main_api.py → scripts/test_system.py
```

### 4. Frontend Changes
```
frontend/ → Browser → Manual testing
```

## Testing Structure

### Unit Tests
- Individual function testing
- Located in each module

### Integration Tests
```
scripts/test_system.py       # Full system integration
scripts/test_recommendations.py  # Recommendation engine
```

### Manual Testing
1. Start server: `python api/main_api.py`
2. Open browser: `http://localhost:8000`
3. Test all form inputs
4. Verify predictions
5. Check accessibility

## Configuration Files

### requirements.txt
```
torch>=1.10.0
fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
```

### .gitignore
```
__pycache__/
*.pyc
*.pth (large model files)
.env
.vscode/
outputs/temp/
```

## Build and Deployment

### Development Build
```bash
python api/main_api.py --reload
```

### Production Build
```bash
uvicorn api.main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0"]
```

## Code Style Guidelines

### Python (PEP 8)
- 4 spaces indentation
- Max line length: 100 characters
- Type hints recommended
- Docstrings for all functions

### JavaScript (ES6+)
- 4 spaces indentation
- camelCase for variables
- PascalCase for classes
- JSDoc comments

### CSS
- BEM methodology
- Mobile-first approach
- CSS variables for theming
- Comments for sections

## Best Practices

### 1. Never Commit
- Large model files (use Git LFS)
- API keys or secrets
- Local configuration files
- Temporary files

### 2. Always Document
- New features
- API changes
- Breaking changes
- Configuration options

### 3. Test Before Push
- Run all tests
- Check code style
- Verify documentation
- Test in multiple browsers

## Migration Guide

### Moving from Old Structure

**Before:**
```
Mobile-Phone-Pricing/
├── README.md
├── CLEANUP_SUMMARY.md
├── test_recommendations.py
└── (everything in root)
```

**After:**
```
Mobile-Phone-Pricing/
├── README.md
├── docs/
│   └── CLEANUP_SUMMARY.md
├── scripts/
│   └── test_recommendations.py
└── (organized structure)
```

## Performance Considerations

### Model Loading
- Models loaded once at startup
- Cached in memory for fast predictions
- Lazy loading for optional features

### API Response Time
- Target: < 100ms per prediction
- Caching: Frequent predictions cached
- Compression: gzip enabled

### Frontend Loading
- CSS: 30KB minified
- JS: 25KB minified
- HTML: 15KB
- Total: ~70KB initial load

## Security Considerations

### Input Validation
- Server-side validation
- Client-side validation (UX)
- Range checks
- Type checking

### API Security
- CORS configured
- Rate limiting
- Input sanitization
- Error message sanitization

## Maintenance

### Regular Tasks
- Update dependencies monthly
- Review and update models quarterly
- Monitor performance metrics
- Update documentation

### Version Control
- Semantic versioning (MAJOR.MINOR.PATCH)
- Changelog maintenance
- Tag releases
- Branch strategy: main → dev → feature

## Contact

For questions about the codebase structure:
- Open an issue on GitHub
- Review existing documentation
- Check inline code comments

---

**Last Updated**: October 23, 2025  
**Version**: 2.0.0  
**Maintainer**: Karthik AK
