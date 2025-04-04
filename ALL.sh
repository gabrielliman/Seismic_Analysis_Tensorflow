#!/bin/bash
readonly GPU_ID=1

./LRP_Parihaka.sh
./LRP_Penobscot.sh
./RPRV_Parihaka.sh
./RPRV_Penobscot.sh
./RPEDS_Parihaka.sh
./RPEDS_Penobscot.sh
./EDS_Parihaka.sh
./EDS_Penobscot.sh