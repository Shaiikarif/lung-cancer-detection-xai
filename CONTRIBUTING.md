# Contributing to PulmoAI

Thank you for your interest in contributing to **PulmoAI**! We welcome contributions from researchers, software engineers, and medical imaging specialists.

## Code of Conduct
Please maintain a respectful, scientific, and collaborative environment.

## Development Workflow

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lung-cancer-detection-xai.git
   cd lung-cancer-detection-xai
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Follow Code Standards:**
   - Adhere strictly to **PEP 8** style guidelines.
   - Include type annotations for all new functions and public methods.
   - Write descriptive docstrings following Google or NumPy format.
   - Add unit tests under `tests/` for any new logic.

5. **Run the Test Suite:**
   ```bash
   python -m unittest discover -s tests -v
   ```
   Ensure all tests pass prior to submitting your Pull Request.

6. **Submit a Pull Request:**
   Provide a concise description of your changes, the rationale, and any experimental validation.
