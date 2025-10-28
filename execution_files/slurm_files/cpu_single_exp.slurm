#!/bin/bash

# Partición en la que se va a ejecutar (cambia a C1 o C2 para CPU)
#SBATCH --partition=C0

# Nombre del trabajo
#SBATCH --job-name=AMGM_nasasrbEx

# Número de tareas (puedes mantener 1 si no usas paralelización)
#SBATCH --ntasks=1

# Número de CPUs a utilizar
#SBATCH --cpus-per-task=32

# Archivo de log donde quedará lo que imprima su software por pantalla
#SBATCH --output=/home/est_posgrado_jose.quiroz/AMGM_test_place/log_outputs/exp%A_%a.log

# Archivo de error donde se guardarán los errores del trabajo
#SBATCH --error=/home/est_posgrado_jose.quiroz/AMGM_test_place/log_error/errors_exp%A_%a.log

# Memoria máxima a utilizar
#SBATCH --mem=0

# Tiempo máximo de ejecución
#SBATCH --time=0

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jose.quiroz@cimat.mx

# Activar el entorno conda
CONDA_PATH=$(which conda)
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
source $CONDA_BASE/etc/profile.d/conda.sh

# Cambiar al directorio de trabajo
cd /home/est_posgrado_jose.quiroz/AMGM_test_place

python experiment_place.py --info 1 --savedir dist5_10 --sys 0 --dist 5 --nexp 10 --conf 2 --iters 10000