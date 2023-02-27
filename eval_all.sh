EXP=chi3d+oldbackbone
python run/validate.py --cfg configs/demo/chi3d.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-s03@best
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-crash@best
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-dance@best

EXP=chi3d+newbackbone
python run/validate.py --cfg configs/demo/chi3d.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-s03@best
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-crash@best
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-dance@best

EXP=chi3d+thuman+oldbackbone
python run/validate.py --cfg configs/demo/chi3d.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-s03@best
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-crash@best
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-dance@best

EXP=chi3d+thuman+newbackbone
python run/validate.py --cfg configs/demo/chi3d.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-s03@best
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-crash@best
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/model_best.pth.tar \
    --out exp-out/$EXP-dance@best

EXP=chi3d+all+thuman+oldbackbone
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/checkpoint_030.pth.tar \
    --out exp-out/$EXP-crash@30
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/checkpoint_030.pth.tar \
    --out exp-out/$EXP-dance@30

EXP=chi3d+all+thuman+newbackbone
python run/demo.py --cfg configs/demo/crash.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/checkpoint_030.pth.tar \
    --out exp-out/$EXP-crash@30
python run/demo.py --cfg configs/demo/dance.yaml \
    --ckpt output/$EXP/chi3d/voxelpose_50/$EXP/checkpoint_030.pth.tar \
    --out exp-out/$EXP-dance@30