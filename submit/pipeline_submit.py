import os
import sys
import getopt
import time

SHELL_NAME = "auto_submit.sh"

def help_info():
    info = [
        "example: pipeline_submit.py -n train -t 10 -g A100 -m 64G",
        "-n <job_name> or --name <job_name>\t\tset job name, will be used as output and error",
        "\t\t\t\t\t\t\t name if they are not defined. Default: default",
        "-o <output_name> or --out <output_name>\t\toutput filename. Default: same as job_name",
        "-e <error_name> or --err <error_name>\t\terror filename. Default: same as job_name",
        "-t <job_time> or --time <job_time>\t\tjob time, in hours. Default: 8",
        "-g <machine_name> or --gpu <machine_name>\tconstraint machine. Default: A100",
        "-m <memory> or --mem <memory>\t\t\tmemory size. Default: 64G",
        "--path <code_path>\t\t\t\ttarget code path. Default: ../frame_pipeline.py",
        "-h or --help\t\t\t\t\thelp message",
    ]
    for i in range(len(info)):
        info[i] = "[HELP] " + info[i]
    return "\n".join(info)


def generate_shell(configs):
    shell_file = open(SHELL_NAME)
    lines = shell_file.readlines()
    shell_file.close()
    
    locate_dic = {      # the line number of each of the settings
        "job_name": 2,
        "output_name": 3,
        "error_name": 4,
        "job_time": 5,
        "gpu": 7,
        "memory": 8,
        "code_path": 22,
    }

    for key in configs:
        loc = locate_dic[key]
        if key in ("job_name", "gpu", "code_path"):
            raw = lines[loc].split("=")
            raw[-1] = configs[key] + "\n"
            lines[loc] = "=".join(raw)
        elif key in ("output_name", "error_name"):
            raw = lines[loc].split("/")
            raw[-1] = configs[key] + "\n"
            lines[loc] = "/".join(raw)
        elif key in ("job_time"):
            raw = lines[loc].split(" ")
            raw[-1] = configs[key] + ":00:00\n"
            lines[loc] = " ".join(raw)
        else:
            raw = lines[loc].split(" ")
            raw[-1] = configs[key] + "\n"
            lines[loc] = " ".join(raw)
    
    new_shell_file = open(SHELL_NAME, "w")
    new_shell_file.writelines(lines)
    new_shell_file.close()


def main(argv):

    configs = {
        "job_name": "default",      # -n, --name
        "output_name" : "",     # -o, --output
        "error_name" : "",      # -e, --error
        "job_time" : "8",    # -t --time
        "gpu" : "A100",         # -g --gpu
        "memory" : "64G",       # -m --memory
        "code_path" : "../frame_pipeline.py",   # --path
    }
    
    try:
        opts, args = getopt.getopt(argv, "hn:o:e:t:g:m:", 
                                   ["help", "name=", "out=", "err=",
                                    "time=", "gpu=", "mem=", "path="])
    except getopt.GetoptError:
        print("[Error]: invalide option or input")
        print("       : pipeline_submit.py -h or --help for infomation")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_info())
            sys.exit()
        elif opt in ("-n", "--name"):
            configs["job_name"] = arg
        elif opt in ("-o", "--out"):
            configs["output_name"] = arg + "_out"
        elif opt in ("-e", "--err"):
            configs["error_name"] = arg + "_err"
        elif opt in ("-t", "--time"):
            configs["job_time"] = arg
        elif opt in ("-g", "--gpu"):
            configs["gpu"] = arg
        elif opt in ("-m", "--memory"):
            configs["memory"] = arg
        elif opt in ("--path"):
            configs["code_path"] = arg

    if configs["output_name"] == "":
        configs["output_name"] = configs["job_name"] + "_out"
    if configs["error_name"] == "":
        configs["error_name"] = configs["job_name"] + "_err"

    print(configs)

    generate_shell(configs)


def execute_shell():
    print(f"$ sbatch {SHELL_NAME}")
    os.system(f"sbatch {SHELL_NAME}")

    time.sleep(1)

    os.system("squeue -u yguan2")


if __name__ == "__main__":
    main(sys.argv[1:])
    execute_shell()