import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PLATFORM = "windows" if sys.platform == "win32" else "linux" if sys.platform == "linux" else ""


def check_version_and_platform() -> bool:
    version = sys.version_info
    return False if version.major != 3 and version.minor < 10 else PLATFORM != ""


def check_git_install() -> None:
    try:
        subprocess.check_call(
            "git --version",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=PLATFORM == "linux",
        )
    except FileNotFoundError:
        print("ERROR: git is not installed, please install git")
        return False
    return True


# windows only
def set_execution_policy() -> None:
    try:
        subprocess.check_call(str(Path("installables/change_execution_policy.bat")))
    except subprocess.SubprocessError:
        try:
            subprocess.check_call(str(Path("installables/change_execution_policy_backup.bat")))
        except subprocess.SubprocessError as e:
            print(f"Failed to change the execution policy with error:\n {e}")
            return False
    return True


def setup_accelerate(platform: str) -> None:
    if platform == "windows":
        path = Path(f"{os.environ['USERPROFILE']}")
    else:
        path = Path.home()
    path = path.joinpath(".cache/huggingface/accelerate/default_config.yaml")
    if path.exists():
        print("Default accelerate config already exists, skipping.")
        return
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open("default_config.yaml", "w") as f:
        f.write("command_file: null\n")
        f.write("commands: null\n")
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("deepspeed_config: {}\n")
        f.write("distributed_type: 'NO'\n")
        f.write("downcase_fp16: 'NO'\n")
        f.write("dynamo_backend: 'NO'\n")
        f.write("fsdp_config: {}\n")
        f.write("gpu_ids: '0'\n")
        f.write("machine_rank: 0\n")
        f.write("main_process_ip: null\n")
        f.write("main_process_port: null\n")
        f.write("main_training_function: main\n")
        f.write("megatron_lm_config: {}\n")
        f.write("mixed_precision: bf16\n")
        f.write("num_machines: 1\n")
        f.write("num_processes: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("tpu_name: null\n")
        f.write("tpu_zone: null\n")
        f.write("use_cpu: false")

    shutil.move("default_config.yaml", str(path.resolve()))


def check_50_series_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        gpu_name = result.stdout.strip()
        if "RTX 50" in gpu_name:
            return True
        elif "NVIDIA" in gpu_name:
            return False
        else:
            return False
    except subprocess.CalledProcessError:
        print("No NVIDIA GPU detected or nvidia-smi not found.")
        return False
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
        return False


def get_torch_version(venv_python: Path) -> str:
    try:
        output = subprocess.check_output(
            [str(venv_python), "-c", "import torch; print(torch.__version__)"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return output.strip()


def get_triton_windows_spec(torch_version: str) -> str:
    # Mapping from triton-lang/triton-windows README compatibility table.
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not match:
        return "triton-windows<3.7"
    major, minor = int(match.group(1)), int(match.group(2))
    if major != 2:
        return "triton-windows<3.7"
    mapping = {
        4: "triton-windows>=3.1,<3.2",
        5: "triton-windows>=3.1,<3.2",
        6: "triton-windows>=3.2,<3.3",
        7: "triton-windows>=3.3,<3.4",
        8: "triton-windows>=3.4,<3.5",
        9: "triton-windows>=3.5,<3.6",
        10: "triton-windows>=3.6,<3.7",
    }
    return mapping.get(minor, "triton-windows<3.7")


def ensure_triton_windows(venv_pip: Path, venv_python: Path) -> None:
    if PLATFORM != "windows":
        return
    torch_version = get_torch_version(venv_python)
    package_spec = get_triton_windows_spec(torch_version)
    print(
        f"Installing Triton for Windows from triton-lang/triton-windows "
        f"(torch={torch_version or 'unknown'}, spec={package_spec})"
    )
    subprocess.run(
        [str(venv_pip), "uninstall", "-y", "triton"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        subprocess.check_call([str(venv_pip), "install", "-U", package_spec])
    except subprocess.CalledProcessError:
        # Conservative fallback noted in triton-windows README.
        subprocess.check_call([str(venv_pip), "install", "-U", "triton-windows<3.7"])


def setup_venv(venv_pip):
    venv_python = Path("venv/Scripts/python.exe" if PLATFORM == "windows" else "venv/bin/python")
    if check_50_series_gpu():
        subprocess.check_call(
            f"{venv_pip} install -U --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128",
            shell=PLATFORM == "linux",
        )
        print("50 series GPU doesn't have xformers support!")
    else:
        subprocess.check_call(
            f"{venv_pip} install -U torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
            shell=PLATFORM == "linux",
        )

        subprocess.check_call(
            f"{venv_pip} install -U xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124",
            shell=PLATFORM == "linux",
        )
    ensure_triton_windows(venv_pip, venv_python)
    if PLATFORM == "windows":
        subprocess.check_call("venv\\Scripts\\python.exe ..\\fix_torch.py")
    subprocess.check_call(f"{venv_pip} install -U -r requirements.txt", shell=PLATFORM == "linux")
    subprocess.check_call(f"{venv_pip} install -U ../custom_scheduler/.", shell=PLATFORM == "linux")
    subprocess.check_call(f"{venv_pip} install -U -r ../requirements.txt", shell=PLATFORM == "linux")
    subprocess.check_call(f"{venv_pip} install -U ../lycoris/.", shell=PLATFORM == "linux")


def sync_sd_scripts_latest():
    try:
        subprocess.check_call("git -C sd_scripts fetch origin --prune", shell=PLATFORM == "linux")
        subprocess.check_call("git -C sd_scripts checkout origin/main", shell=PLATFORM == "linux")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update sd_scripts to origin/main with error:\n {e}")
        return False
    return True


# colab only
def setup_colab(venv_pip):
    setup_venv(venv_pip)
    setup_accelerate("linux")


def ask_yes_no(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ")
    return reply == "y"


def setup_config(colab: bool = False, local: bool = False) -> None:
    if colab:
        config = {
            "remote": True,
            "remote_mode": "cloudflared",
            "kill_tunnel_on_train_start": True,
            "kill_server_on_train_end": True,
            "colab": True,
            "port": 8000,
        }
        with open("config.json", "w") as f:
            f.write(json.dumps(config, indent=2))
        return
    is_remote = False if local else ask_yes_no("are you using this remotely?")
    remote_mode = "none"
    if is_remote:
        remote_mode = "ngrok" if ask_yes_no("do you want to use ngrok?") else "cloudflared"
    ngrok_token = ""
    if remote_mode == "ngrok":
        ngrok_token = input(
            "copy paste your token from your ngrok dashboard (https://dashboard.ngrok.com/get-started/your-authtoken) (requires account): "
        )

    with open("config.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "remote": is_remote,
                    "remote_mode": remote_mode,
                    "ngrok_token": ngrok_token,
                    "port": 8000,
                },
                indent=2,
            )
        )


def main():
    if not check_version_and_platform() or not check_git_install():
        quit()

    subprocess.check_call("git submodule sync --recursive", shell=PLATFORM == "linux")
    subprocess.check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")
    if not sync_sd_scripts_latest():
        quit()

    if PLATFORM == "windows":
        print("setting execution policy to unrestricted")
        if not set_execution_policy():
            quit()

    setup_config(
        len(sys.argv) > 1 and sys.argv[1] == "colab",
        len(sys.argv) > 1 and sys.argv[1] == "local",
    )

    os.chdir("sd_scripts")
    if PLATFORM == "windows":
        pip = Path("venv/Scripts/pip.exe")
    else:
        pip = Path("venv/bin/pip")

    print("creating venv and installing requirements")
    subprocess.check_call(f"{sys.executable} -m venv venv", shell=PLATFORM == "linux")

    if len(sys.argv) > 1 and sys.argv[1] == "colab":
        setup_colab(pip)
        print("completed installing")
        quit()

    setup_venv(pip)
    setup_accelerate(PLATFORM)

    print("Completed installing, you can run the server via the run.bat or run.sh files")


if __name__ == "__main__":
    main()
