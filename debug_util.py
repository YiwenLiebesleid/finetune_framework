import time

debug = True
prefix = "[DEBUG]"

def debug_prt(msg, msg_name=""):
    msg = str(msg)
    if debug:
        output_msg = []
        now = time.ctime().split(" ")[-2]
        output_msg.append(prefix)
        output_msg.append(now)
        if len(msg_name) > 0:
            output_msg.append(msg_name + ":")
        output_msg.append(msg)
        print(" ".join(output_msg))

if __name__ == "__main__":
    num = 6
    debug_prt(num)