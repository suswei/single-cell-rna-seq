#First, create Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv based on Pbmc_CellName_Label_GeneCount.csv, Pbmc_Batch.csv, 
#Pbmc_GeneInfo.csv. The three files datasetsPbmc_CellName_Label_GeneCount.csv, Pbmc_Batch.csv, Pbmc_GeneInfo.csv are created 
#by code in break_SCVI_generate_artificial_dataset.py

setwd("D:/UMelb/PhD_Projects/Project1_Modify_SCVI/")
library(data.table)
library(scater)

Pbmc_CellName_Label_GeneCount = fread("./data/break_SCVI/Pbmc_CellName_Label_GeneCount.csv")
Pbmc_Batch = fread("./data/break_SCVI/Pbmc_batch.csv")
Pbmc_GeneInfo = fread("./data/break_SCVI/gene_info_pbmc.csv")

colnames(Pbmc_CellName_Label_GeneCount)[3:dim(Pbmc_CellName_Label_GeneCount)[2]] = Pbmc_GeneInfo$ENSG

Pbmc_CellName_Label_Batch_GeneCount = cbind(Pbmc_Batch,Pbmc_CellName_Label_GeneCount)

Pbmc_CellName_Label_Batch_GeneCount = Pbmc_CellName_Label_Batch_GeneCount[,c(2:3,1,4:dim(Pbmc_CellName_Label_Batch_GeneCount)[2]),with=F]

Pbmc_GeneCount_Matrix = t(as.matrix(Pbmc_CellName_Label_Batch_GeneCount[,4:dim(Pbmc_CellName_Label_Batch_GeneCount)[2]])) 
##Column is cell, row is gene

Pbmc_Info = Pbmc_CellName_Label_Batch_GeneCount[,1:3]
##Pbmc cell info

Sce = SingleCellExperiment(
  assays = list(counts = Pbmc_GeneCount_Matrix), 
  colData = Pbmc_Info
)

Sce2 = calculateQCMetrics(Sce)

Sce2_CellMetrics = as.data.frame(colData(Sce2)[,c("total_counts","total_features_by_counts", "pct_counts_in_top_100_features")])

Pbmc_CellName_Label_Batch_CellMetric_GeneCount = cbind(Sce2_CellMetrics, Pbmc_CellName_Label_Batch_GeneCount)

Cell_Types = c('B cells', 'CD14+ Monocytes', 'CD4 T cells', 'CD8 T cells','Dendritic Cells', 'FCGR3A+ Monocytes', 'Megakaryocytes', 'NK cells', 'Other')

Cell_Type_Function = function(x){
  return(Cell_Types[x+1])
}
Cell_Type_Function = Vectorize(Cell_Type_Function)

Pbmc_CellName_Label_Batch_CellMetric_GeneCount$Cell_Types = Cell_Type_Function(Pbmc_CellName_Label_Batch_CellMetric_GeneCount$Labels)
Pbmc_CellName_Label_Batch_CellMetric_GeneCount = Pbmc_CellName_Label_Batch_CellMetric_GeneCount[,c(1:3,dim(Pbmc_CellName_Label_Batch_CellMetric_GeneCount)[2],4:(dim(Pbmc_CellName_Label_Batch_CellMetric_GeneCount)[2]-1))]
write.table(Pbmc_CellName_Label_Batch_CellMetric_GeneCount, file="./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv",row.names = F, col.names = T, sep=",")


#Second, get exploratory analysis for original PBMC count data

library(ggplot2)

Pbmc_dataset = fread("./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv")

#get the boxplots of library size, number of genes expressed between batch 0 and batch 1, and between batch 0 and batch 1 in every cell type
png("./result/break_SCVI/LibrarySize_By_Batch.png")
p1 = ggplot(aes(y = total_counts, x = factor(batchpbmc4k)), data = Pbmc_dataset) + geom_boxplot() + labs(y = 'Library Size', x='batch')
print(p1)
dev.off()
png("./result/break_SCVI/GeneExpressed_By_Batch.png")
p2 = ggplot(aes(y = total_features_by_counts, x = factor(batchpbmc4k)), data = Pbmc_dataset) + geom_boxplot() +
  labs(y = 'Number of Genes with Non-Zero Count', x = 'Batch')
print(p2)
dev.off()
png("./result/break_SCVI/LibrarySize_By_CellTypes_Batch.png")
p3 = ggplot(aes(y = total_counts, x =Labels, fill = factor(batchpbmc4k)), data = Pbmc_dataset) + geom_boxplot() + 
  labs(y = "Library Size", x = 'Cell Type' , caption = "0: B cells, 1: CD14+ Monocytes, 2: CD4 T cells,\n3:  CD8 T cells, 4:  Dendritic Cells, 5: FCGR3A+ Monocytes,\n6: Megakaryocytes, 7:  NK cells, 8: Other") + 
  theme(plot.caption = element_text(hjust = 0))+ guides(fill=guide_legend(title='Batch'))
print(p3)
dev.off()
png("./result/break_SCVI/GeneExpressed_By_CellTypes_Batch.png")
p4 = ggplot(aes(y = total_features_by_counts, x =Labels, fill = factor(batchpbmc4k)), data = Pbmc_dataset) + geom_boxplot()+
  labs(y = "Number of Genes with Non-Zero Count", x='Cell Type', caption = "0: B cells, 1: CD14+ Monocytes, 2: CD4 T cells,\n3:  CD8 T cells, 4:  Dendritic Cells, 5: FCGR3A+ Monocytes,\n6: Megakaryocytes, 7:  NK cells, 8: Other") +
  theme(plot.caption = element_text(hjust = 0)) + guides(fill=guide_legend(title='Batch'))
print(p4)
dev.off()


#Change library size
Pbmc_dataset = fread("./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv")
Pbmc_dataset = Pbmc_dataset[,4:dim(Pbmc_dataset)[2]]

times = c(1/10, 3/10, 3, 10)

##8030/11990 cells are from batch 0, 3960/11990 cells are from batch1
for (i in 1:length(times)){
  
  for(j in 1:2){
    Pbmc_dataset1 = Pbmc_dataset
    
    for(k in 1:nrow(Pbmc_dataset1)){
      if(Pbmc_dataset1$batchpbmc4k[k]==j-1){
        total_count = rowSums(Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])
        prob_vector = as.numeric((Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])/total_count)
        Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]] = data.frame(matrix(rmultinom(1, size=total_count*times[i], prob=prob_vector),nrow=1))
      }
    }
    
    Pbmc_GeneCount_Matrix = t(as.matrix(Pbmc_dataset1[,5:dim(Pbmc_dataset1)[2]])) 
    #Column is cell, row is gene
    
    Pbmc_Info = Pbmc_dataset1[,1:4]
    #Pbmc cell info
    
    Sce = SingleCellExperiment(
      assays = list(counts = Pbmc_GeneCount_Matrix), 
      colData = Pbmc_Info
    )
    
    Sce2 = calculateQCMetrics(Sce)
    
    Sce2_CellMetrics = as.data.frame(colData(Sce2)[,c("total_counts","total_features_by_counts", "pct_counts_in_top_100_features")])
    
    Pbmc_CellName_Label_Batch_CellMetric_GeneCount = cbind(Sce2_CellMetrics, Pbmc_dataset1)
    
    Outpath = paste0("./data/break_SCVI/Change_Library_Size/ModifyBatch",j-1,"_ratio",times[i],".csv")
    write.table(Pbmc_CellName_Label_Batch_CellMetric_GeneCount,file=Outpath, row.names = F, col.names = T, sep=",")
  }
}


#change the number of gene expressed

Pbmc_dataset = fread("./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv")
Pbmc_dataset = Pbmc_dataset[,4:dim(Pbmc_dataset)[2]]
times = c(1/10,3/10,4/5)
for(i in 1:length(times)){
  for(j in 1:2){
    Pbmc_dataset1 = Pbmc_dataset
    for(k in 1:nrow(Pbmc_dataset1)){
      if(Pbmc_dataset1$batchpbmc4k[k]==j-1){
        total_count = rowSums(Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])
        gene_count_onecell = as.numeric(Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])
        index_expressedgene = which(gene_count_onecell>0)
        random_index = sample(x=1:length(index_expressedgene),size=round(length(index_expressedgene)*(1-times[i])),replace=F)
        index_to_set_zero = index_expressedgene[random_index]
        gene_count_onecell[index_to_set_zero] = 0
        
        prob_vector_modified = gene_count_onecell/sum(gene_count_onecell)
        Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]] = data.frame(matrix(rmultinom(1, size=total_count, prob=prob_vector_modified),nrow=1))
      }
    }
    
    Pbmc_GeneCount_Matrix = t(as.matrix(Pbmc_dataset1[,5:dim(Pbmc_dataset1)[2]])) 
    #Column is cell, row is gene
    
    Pbmc_Info = Pbmc_dataset1[,1:4]
    #Pbmc cell info
    
    Sce = SingleCellExperiment(
      assays = list(counts = Pbmc_GeneCount_Matrix), 
      colData = Pbmc_Info
    )
    
    Sce2 = calculateQCMetrics(Sce)
    
    Sce2_CellMetrics = as.data.frame(colData(Sce2)[,c("total_counts","total_features_by_counts", "pct_counts_in_top_100_features")])
    
    Pbmc_CellName_Label_Batch_CellMetric_GeneCount = cbind(Sce2_CellMetrics, Pbmc_dataset1)
    
    Outpath = paste0("./data/break_SCVI/Change_Expressed_Gene_Number/ModifyBatch",j-1,"_ratio",times[i],".csv")
    write.table(Pbmc_CellName_Label_Batch_CellMetric_GeneCount,file=Outpath, row.names = F, col.names = T, sep=",")
  }
}


# change gene expression proportion

Pbmc_dataset = fread("./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv")
Pbmc_dataset = Pbmc_dataset[,4:dim(Pbmc_dataset)[2]]

#change the proportion of gene expressed. My original plan changes the proportion too much, even the cell's 
#expression architecture is changed. Therefore, modify the code according to 2019-04-26_meeting_summary.doc to 
#generate dataset with some genes' expression proportion changed while reserving the overall expression
#architecture

Pbmc_dataset_GeneCount = Pbmc_dataset[,5:dim(Pbmc_dataset)[2]]
Pbmc_dataset_GeneCount_Logic = (Pbmc_dataset_GeneCount==0)
Zeros_Vector = colSums(Pbmc_dataset_GeneCount_Logic)
length(which(Zeros_Vector==0))
#only 9 genes are expressed in all cells, therefore I need to loosen the criteria that I only change the
#genes that are expressed in all cells.

Zeros_Percentage = Zeros_Vector/dim(Pbmc_dataset_GeneCount)[1]
sum(Zeros_Percentage<0.5)
#only 58 genes are expressed in 80% cells, only 117 genes are expressed in 60% cells, 168 genes are expressed in 50% cells
#all these 58, 117 and 168 genes have median expression in the first 200 places. It can be checked

index = which(Zeros_Percentage<0.5)
LessZero_Gene = colnames(Pbmc_dataset_GeneCount)[index]

Gene_MeanExpression = colMedians(as.matrix(Pbmc_dataset_GeneCount))
Gene_MeanExpression_Dataset = data.table(GeneName = colnames(Pbmc_dataset_GeneCount),Gene_Median = Gene_MeanExpression) 
Gene_MeanExpression_Dataset_Ordered = Gene_MeanExpression_Dataset[order(Gene_Median,decreasing=T),]
MostExpressed_200 = Gene_MeanExpression_Dataset_Ordered$GeneName[1:200]

#I decide to change 58*1/4 genes's expression proportion which are expressed in 80% cells and median expressions are in the first 200
#places by a certain ratio, change 117*1/4 genes which are expressed in 60% cells and median expressions are in the first 200
#places, and change the 168*1/4 genes which are expressed in 50% cells and median expressions are in the first 200
#places. The left genes in the 58 genes, or 117 genes, or 168 genes are rescaled. and all the other genes except the
#58, 117 or 168 genes are unchanged.

Proportions = c(0.2,0.4,0.5)
FCs = c( 1/10, 3/10, 3, 10)
for(i in 1:length(Proportions)){
  for(j in 1:length(FCs)){
    for(m in 1:2){
      index = which(Zeros_Percentage<Proportions[i])
      LessZero_Gene = colnames(Pbmc_dataset_GeneCount)[index]
      Genes_List = LessZero_Gene[LessZero_Gene %in% MostExpressed_200]
      Genes_List_Median_Dataset = Gene_MeanExpression_Dataset_Ordered[Gene_MeanExpression_Dataset_Ordered$GeneName %in% Genes_List,]
      
      
      To_Change_Genes_List = Genes_List_Median_Dataset$GeneName[floor(length(Genes_List)*3/4):length(Genes_List)]
      To_Scale_Genes_List = Genes_List[1:(floor(length(Genes_List)*3/4)-1)]
      Pbmc_dataset1 = Pbmc_dataset
      for(k in 1:nrow(Pbmc_dataset1)){
        if(Pbmc_dataset1$batchpbmc4k[k]==m-1){
          total_count = rowSums(Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])
          gene_count_onecell = as.numeric(Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]])
          
          prob_vector = gene_count_onecell/total_count
          prob_dataset = data.table(geneName=colnames(Pbmc_dataset1)[5:ncol(Pbmc_dataset1)], prob=prob_vector)
          prob_dataset$prob[which(prob_dataset$geneName %in% To_Change_Genes_List)] = prob_dataset$prob[which(prob_dataset$geneName %in% To_Change_Genes_List)]*FCs[j]
          
          Ratio_For_Scale = (sum(prob_dataset$prob[which(prob_dataset$geneName %in% To_Scale_Genes_List)])-(sum(prob_dataset$prob)-1))/sum(prob_dataset$prob[which(prob_dataset$geneName %in% To_Scale_Genes_List)])        
          prob_dataset$prob[which(prob_dataset$geneName %in% To_Scale_Genes_List)] = prob_dataset$prob[which(prob_dataset$geneName %in% To_Scale_Genes_List)]* Ratio_For_Scale
          
          Pbmc_dataset1[k,5:dim(Pbmc_dataset1)[2]] = data.frame(matrix(rmultinom(1, size=total_count, prob=prob_dataset$prob),nrow=1))
        }
      }
      Pbmc_GeneCount_Matrix = t(as.matrix(Pbmc_dataset1[,5:dim(Pbmc_dataset1)[2]])) 
      #Column is cell, row is gene
      
      Pbmc_Info = Pbmc_dataset1[,1:4]
      #Pbmc cell info
      
      Sce = SingleCellExperiment(
        assays = list(counts = Pbmc_GeneCount_Matrix), 
        colData = Pbmc_Info
      )
      
      Sce2 = calculateQCMetrics(Sce)
      
      Sce2_CellMetrics = as.data.frame(colData(Sce2)[,c("total_counts","total_features_by_counts", "pct_counts_in_top_100_features")])
      
      Pbmc_CellName_Label_Batch_CellMetric_GeneCount = cbind(Sce2_CellMetrics, Pbmc_dataset1)
      
      Outpath = paste0("./data/break_SCVI/Change_Gene_Expression_Proportion/ModifyProportion",Proportions[i],"_Batch",m-1, "_ratio",FCs[j] , ".csv")
      write.table(Pbmc_CellName_Label_Batch_CellMetric_GeneCount,file=Outpath, row.names = F, col.names = T, sep=",")
    }
  }
}

