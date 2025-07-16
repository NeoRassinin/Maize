import subprocess
from pathlib import Path


def run_streamlit_app(script_path: str = "Streamlit_app.py") -> None:
    """
    Запускает Streamlit-приложение через subprocess.

    Args:
        script_path (str): Путь к файлу с приложением Streamlit.
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Файл {script} не найден.")

    subprocess.run(["streamlit", "run", str(script)], check=True)


if __name__ == "__main__":
    run_streamlit_app()
