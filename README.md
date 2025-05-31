# Context Window Calculator

A Python application to calculate context window sizes, particularly for use with vLLM back-ends and Hugging Face (HF) Transformer models. Understanding context window limitations is crucial for optimizing the performance of these large language models. This project appears to use Streamlit for the user interface.

## Description

This project provides functionality to calculate context window information relevant for vLLM deployments. It helps in determining appropriate input lengths for various HF Transformer models that are beneficial to or commonly used with vLLM. The main application logic seems to be in `app.py` and core calculations in `calculator_logic.py`.

## Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application (assuming it's a Streamlit app based on the running process):

```bash
streamlit run app.py
```

## Project Structure

-   `app.py`: Main application file (likely Streamlit).
-   `calculator_logic.py`: Contains the core logic for calculations.
-   `requirements.txt`: Lists project dependencies.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)