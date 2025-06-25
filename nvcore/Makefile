# Makefile for QUSIM NV Center Simulation Documentation
# Supports both Sphinx documentation generation and general project tasks

.PHONY: help docs docs-clean docs-serve test clean all install

# Default target
help:
	@echo "QUSIM NV Center Simulation - Makefile"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  docs         Build Sphinx HTML documentation"
	@echo "  docs-clean   Clean documentation build directory"
	@echo "  docs-serve   Build docs and serve on http://localhost:8000"
	@echo "  test         Run all tests"
	@echo "  clean        Clean Python cache files"
	@echo "  install      Install Python dependencies"
	@echo "  all          Build docs and run tests"
	@echo ""
	@echo "Documentation targets:"
	@echo "  make docs              # Build HTML docs"
	@echo "  make docs-serve        # Build and serve docs"
	@echo "  make docs-clean        # Clean docs build"
	@echo ""
	@echo "Development targets:"
	@echo "  make test              # Run test suite"
	@echo "  make clean             # Clean cache files"
	@echo "  make install           # Install dependencies"
	@echo ""
	@echo "Experiment targets:"
	@echo "  make experiment-minimal # Minimal π-pulse demo (fastest)"
	@echo "  make experiment-quick   # Quick π-pulse demo"
	@echo "  make experiment-full    # Full readout experiment"
	@echo "  make experiment-help    # Show experiment details"

# Documentation targets
docs: docs-clean
	@echo "Building Sphinx documentation..."
	@echo "================================="
	@mkdir -p ../docs/_build/html
	@cd .. && python -m sphinx -b html -E -a docs docs/_build/html -q
	@echo ""
	@echo "✅ Documentation built successfully!"
	@echo "📖 Open: docs/_build/html/index.html"
	@echo "🌐 Or run: make docs-serve"

docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf ../docs/_build/
	@rm -rf ../docs/_doctrees/
	@echo "✅ Documentation cleaned"

docs-serve: docs
	@echo ""
	@echo "🌐 Starting QUSIM Documentation Server..."
	@echo "📖 Documentation: http://localhost:8000" 
	@echo "Press Ctrl+C to stop the server"
	@echo ""
	@cd .. && python serve_docs.py

# Test targets
test:
	@echo "Running QUSIM test suite..."
	@echo "============================"
	@python test.py
	@echo ""
	@echo "✅ Tests completed"

test-verbose:
	@echo "Running QUSIM test suite (verbose)..."
	@echo "======================================"
	@python test.py --verbose
	@echo ""
	@echo "✅ Verbose tests completed"

# CLI targets
demo:
	@echo "Running QUSIM demo..."
	@echo "====================="
	@python core.py --demo --time 1e-6 --noise c13_bath

benchmark:
	@echo "Running QUSIM benchmark..."
	@echo "=========================="
	@python core.py --benchmark

# Experiment targets
experiment-minimal:
	@echo "Running minimal π-pulse demo..."
	@echo "==============================="
	@cd ../experiments && python minimal_demo.py

experiment-quick:
	@echo "Running quick π-pulse experiment..."
	@echo "==================================="
	@cd ../experiments && python quick_demo.py

experiment-full:
	@echo "Running full π-pulse readout experiment..."
	@echo "=========================================="
	@cd ../experiments && python pi_pulse_readout.py

experiment-help:
	@echo "Available Experiments:"
	@echo "====================="
	@echo "  make experiment-minimal  # Minimal π-pulse demo (fastest)"
	@echo "  make experiment-quick    # Quick π-pulse demo (fast)"
	@echo "  make experiment-full     # Complete readout experiment"
	@echo ""
	@echo "Manual execution:"
	@echo "  cd ../experiments"
	@echo "  python minimal_demo.py   # Minimal demo (no simulation)"
	@echo "  python quick_demo.py     # Quick demo (simplified)"
	@echo "  python pi_pulse_readout.py  # Full experiment (realistic)"

# Clean targets
clean:
	@echo "Cleaning Python cache files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete
	@rm -rf .pytest_cache/
	@echo "✅ Python cache cleaned"

clean-all: clean docs-clean
	@echo "✅ All artifacts cleaned"

# Installation targets
install:
	@echo "Installing Python dependencies..."
	@echo "================================="
	@pip install -r ../requirements.txt
	@echo "✅ Dependencies installed"

install-dev: install
	@echo "Installing development dependencies..."
	@pip install sphinx sphinx-rtd-theme pytest coverage
	@echo "✅ Development dependencies installed"

# Comprehensive targets
all: clean test docs
	@echo ""
	@echo "🎉 All tasks completed successfully!"
	@echo "✅ Tests passed"
	@echo "✅ Documentation built"
	@echo ""
	@echo "Next steps:"
	@echo "  make docs-serve    # View documentation"
	@echo "  make demo          # Run demo"
	@echo "  make benchmark     # Performance test"

# Development workflow
dev: install-dev clean test docs
	@echo ""
	@echo "🚀 Development environment ready!"
	@echo "================================="
	@echo ""
	@echo "Quick commands:"
	@echo "  python core.py --help     # CLI help"
	@echo "  python core.py --demo     # Run demo"
	@echo "  python test.py            # Run tests"
	@echo "  make docs-serve           # View docs"

# Documentation with autobuild (if sphinx-autobuild is available)
docs-auto:
	@echo "Starting Sphinx autobuild..."
	@echo "=============================="
	@if command -v sphinx-autobuild >/dev/null 2>&1; then \
		cd .. && sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000; \
	else \
		echo "❌ sphinx-autobuild not found. Install with:"; \
		echo "   pip install sphinx-autobuild"; \
		echo ""; \
		echo "Using regular build instead..."; \
		make docs-serve; \
	fi

# Release preparation
release-check: clean test docs
	@echo "Release readiness check..."
	@echo "=========================="
	@echo "✅ Tests passed"
	@echo "✅ Documentation built"
	@echo ""
	@echo "Manual checks required:"
	@echo "  - Review CHANGELOG.md"
	@echo "  - Update version numbers"
	@echo "  - Test on clean environment"

# Help for specific components
help-docs:
	@echo "Documentation Help"
	@echo "=================="
	@echo ""
	@echo "Building documentation:"
	@echo "  make docs              # Build HTML documentation"
	@echo "  make docs-serve        # Build and serve on localhost:8000"
	@echo "  make docs-clean        # Clean build artifacts"
	@echo "  make docs-auto         # Auto-rebuild on changes (needs sphinx-autobuild)"
	@echo ""
	@echo "Documentation structure:"
	@echo "  docs/index.rst         # Main documentation page"
	@echo "  docs/api_reference.rst # API documentation"
	@echo "  docs/conf.py           # Sphinx configuration"
	@echo "  docs/_build/html/      # Generated HTML output"
	@echo ""
	@echo "Dependencies:"
	@echo "  sphinx                 # Documentation generator"
	@echo "  sphinx-rtd-theme       # Read the Docs theme"
	@echo "  myst-parser            # Markdown support (optional)"

help-test:
	@echo "Testing Help"
	@echo "============"
	@echo ""
	@echo "Running tests:"
	@echo "  make test              # Run all tests"
	@echo "  make test-verbose      # Run with verbose output"
	@echo "  python test.py         # Direct test execution"
	@echo ""
	@echo "Test categories:"
	@echo "  - Spin operator tests"
	@echo "  - Hamiltonian tests"
	@echo "  - Noise generator tests"
	@echo "  - Integration tests"
	@echo "  - Performance benchmarks"