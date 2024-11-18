mkdir ${SLURM_TMPDIR}/torch.sif && tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch.tar -C $SLURM_TMPDIR;

cd ../

singularity shell --nv --writable --bind $PWD:/home/mozaffar --bind $SLURM_TMPDIR:/tmp $SLURM_TMPDIR/torch.sif