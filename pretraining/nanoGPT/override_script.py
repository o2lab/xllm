import os
try:
    flag = False
    for key, value in os.environ.items():
        if key == "SLURM_JOB_NODELIST":
            print(key, value)
            ins_type = value[0]
            nodes = value[2:-1]
            # value example: 123,456    045-047
            master, worker = nodes[:3], nodes[-3:]
            print("Master: ", master)
            print("Worker: ", worker)

            master = ins_type + master
            worker = ins_type + worker
            with open("/scratch/user/siweicui/nanoGPT/master.sh", 'w') as f:
                f.write("torchrun  --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=")
                f.write(master)
                f.write(" --master_port=1234  train.py config/train_gpt2.py")
            with open("/scratch/user/siweicui/nanoGPT/worker.sh", "w") as f:
                f.write("torchrun  --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=")
                f.write(master)
                f.write(" --master_port=1234  train.py config/train_gpt2.py")
            
            # generate worker script
            with open("/scratch/user/siweicui/nanoGPT/run_worker.sh", "w") as f:
                # f.write("ssh ")
                # f.write(worker+"\n")
                f.write("source activate llm\n")
                f.write("cd /scratch/user/siweicui/nanoGPT/\n")
                f.write("nohup ./worker.sh > worker.log 2>&1 &")

            with open("/scratch/user/siweicui/nanoGPT/run_master.sh", "w") as f:
                f.write("ssh ")
                f.write(worker+" \"/scratch/user/siweicui/nanoGPT/run_worker.sh\"\n")
                f.write("cd /scratch/user/siweicui/nanoGPT/\n")
                f.write("echo \"Running Master Jobs\"\n")
                f.write("./master.sh")
            print("OVERRIDE: Success")
            flag = True
    if not flag:
        print("OVERRIDE: Not found.")
    # print(get("SLURM_JOB_NODELIST", "c[476,492]"))

except:
    print("OVERRIDE: Failed")
    pass