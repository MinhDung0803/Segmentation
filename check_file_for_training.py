import os

def create_folders_for_training():
    if not os.path.exists("./runs"):
        os.mkdir("./runs")
        os.mkdir("./runs/mhpv1")
        os.mkdir("./runs/mhpv1/icnet")
        os.mkdir("./runs/mhpv1/icnet/resnet50")
    elif not os.path.exists("./runs/mhpv1"):
        os.mkdir("./runs/mhpv1")
        os.mkdir("./runs/mhpv1/icnet")
        os.mkdir("./runs/mhpv1/icnet/resnet50")
    elif not os.path.exists("./runs/mhpv1/icnet"):
        os.mkdir("./runs/mhpv1/icnet")
        os.mkdir("./runs/mhpv1/icnet/resnet50")
    elif not os.path.exists("./runs/mhpv1/icnet/resnet50"):
        os.mkdir("./runs/mhpv1/icnet/resnet50")

