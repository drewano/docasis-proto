# Contributing to RAG Intelligent Agent

Thank you for considering contributing to the RAG Intelligent Agent! This document outlines the process for contributing to the project and the standards we follow.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue has already been reported in the [Issues](https://github.com/yourusername/rag-intelligent-agent/issues) section
2. If not, create a new issue with a descriptive title and detailed information:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - Your environment (OS, Python version, etc.)

### Making Changes

1. Fork the repository
2. Create a new branch from `main` with a descriptive name:
   ```
   git checkout -b feature/your-feature-name
   ```
   or
   ```
   git checkout -b fix/issue-you-are-fixing
   ```
3. Make your changes, following the coding standards
4. Write or update tests to cover your changes
5. Ensure all tests pass
6. Update documentation to reflect your changes

### Pull Request Process

1. Update the README.md or relevant documentation with details of your changes
2. Ensure your code follows the project's coding standards
3. Make sure all tests pass
4. Submit a pull request to the `main` branch
5. Describe what your PR does and reference any related issues

## Development Setup

1. Clone your forked repository:
   ```
   git clone https://github.com/yourusername/rag-intelligent-agent.git
   cd rag-intelligent-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration

5. Run tests to ensure everything is working:
   ```
   pytest
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep functions and methods focused on a single responsibility
- Maximum line length is 100 characters

### Commenting

- Write clear, concise comments explaining "why" not "what"
- Keep comments up-to-date with code changes
- Use TODO comments for code that is temporary or needs improvement

### Testing

- Write tests for all new functionality
- Maintain test coverage at or above 80%
- Structure tests using the Arrange-Act-Assert pattern
- Use descriptive test names that explain what is being tested

## Directory Structure

Maintain the established project structure:

```
.
├── src/                    # Source code
│   ├── models/             # Model-related code
│   ├── data/               # Data processing utilities
│   ├── utils/              # Helper functions
│   ├── api/                # API interfaces
│   └── ui/                 # Streamlit UI code
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Documentation

- Update documentation for any changes you make
- Document public API methods and classes
- Keep the README.md up-to-date
- Add examples for new functionality

## Review Process

- All contributions require review before being merged
- Address all comments and requested changes
- A maintainer will merge your PR once it's approved

## License

By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to RAG Intelligent Agent! 