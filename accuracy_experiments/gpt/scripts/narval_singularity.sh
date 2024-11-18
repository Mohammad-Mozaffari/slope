cd ../

module load apptainer

rm -rf $SLURM_TMPDIR/torch-gpt.sif;
mkdir ${SLURM_TMPDIR}/torch-gpt.sif;
tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch-gpt.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls/certs/ca-bundle.crt;
singularity shell --bind $PWD:/home/mozaffar --bind $SLURM_TMPDIR:/tmp --nv ${SLURM_TMPDIR}/torch-gpt.sif