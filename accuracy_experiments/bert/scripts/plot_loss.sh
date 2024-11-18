cd ../

for PHASE in 1 2
do
    if [[ $PHASE == 1 ]];
    then
        # #Phase 1 Settings
        TITLE="BERT-Large-Uncased_Pretraining_Phase_1"
        OUTPUT_DIR=bert_phase1.pdf
        FILE_LIST=phase1-fused-lamb
        LEGEND_LIST="Dense"
        FILE_LIST+=",phase1-fused-lamb-pruned-static-weight-8nodes"
        LEGEND_LIST+=",2:4_Sparsity"
        FILE_LIST+=",phase1-fused-lamb-pruned-static-weight-sparsity_increment12-8nodes"
        LEGEND_LIST+=",Mixed_2:4_and_4:8_Sparsity"
    fi
    if [[ $PHASE == 2 ]];
    then
        #Phase 2 Settings
        TITLE="BERT-Large-Uncased_Pretraining_Phase_2"
        OUTPUT_DIR=bert_phase2.pdf
        FILE_LIST=phase2-fused-lamb
        LEGEND_LIST="Dense"
        FILE_LIST+=",phase2-fused-lamb-pruned-static-weight-8nodes"
        LEGEND_LIST+=",2:4_Sparsity"
        FILE_LIST+=",phase2-fused-lamb-pruned-static-weight-sparsity_increment12-8nodes"
        LEGEND_LIST+=",Mixed_2:4_and_4:8_Sparsity"
    fi
    CMD="python utils/plot.py --file_list ${FILE_LIST} --legend_list ${LEGEND_LIST} --title "${TITLE}" --output_dir ${OUTPUT_DIR}"
    echo $CMD
    $CMD
done