tableVIC <- read.csv("~/data/simulation/tableVIC.csv")
library(ggplot2)

model_splitted = strsplit(tableVIC$model, "_")
tableVIC["model_delta"] = unlist(lapply(model_splitted, function(x){
  if(is.element("AR", x)){ret = "R"}
  if(is.element("ART", x)){ret = "[-1,1]"}
  if(is.element("RW", x)){ret = "1"}
  return(ret)
}))
tableVIC["model_phicovm"] = unlist(lapply(model_splitted, function(x){
  if(is.element("MVnormal", x)){ret = "general"}
  if(is.element("normal", x)){ret = "diagonal"}
  return(ret)
}))
tableVIC[,"true_delta"] = sapply(tableVIC[,"delta"], 
                                 function(x){switch(x, "zero" = 0, "half" = 0.5, "one" = 1)})

# notDPF = tableVIC[tableVIC$model != "DPF_RW_normal",]


ggplot(tableVIC, aes(x=true_delta, y=VAIC))+
  geom_line(aes(color = model))+
  scale_color_manual(breaks = unique(tableVIC$model),
                     values=c("red", "orange", 
                              "blue", "skyblue",
                              "darkgreen", "yellowgreen","black"))+
  facet_wrap(~T, scales="free_y")+
  theme_minimal()

data <- read.csv("~/data/simulation/data_tableVIC.csv")
data$Model = sapply(data[,"model"], 
                    function(x){switch(x, 
                                       "AR_MVnormal" = "A", 
                                       "AR_normal" = "B", 
                                       "RW_MVnormal" = "C",
                                       "RW_normal" = "D",
                                       "ART_MVnormal" = "E",
                                       "ART_normal" = "F",
                                       "DPF_RW_normal" = "G")})
data[,"true_delta"] = sapply(data[,"delta"], 
                             function(x){switch(x, "zero" = 0, "half" = 0.5, "one" = 1)})
data$setting = apply(data, 1, function(x){
  paste0("T = ", x["T"], ", delta = ", x["true_delta"])
})

cairo_pdf("~/TBIP_colab/TVPF/simulation/data/boxplot_VAIC.pdf",
          width = 8, height = 5.5)
{
ggplot(data, aes(x=Model, y=VAIC/1000))+
  geom_boxplot(aes(fill = Model), show.legend = FALSE, coef = Inf)+
  scale_fill_manual(values=c("red", "orange", 
                             "blue", "skyblue",
                             "darkgreen", "yellowgreen","black"))+
  facet_wrap(~setting, scales="free_y")+
  theme_minimal()
}
dev.off()

cairo_pdf("~/TBIP_colab/TVPF/simulation/data/boxplot_VBIC.pdf",
          width = 8, height = 5.5)
{
  ggplot(data, aes(x=Model, y=VBIC/1000))+
    geom_boxplot(aes(fill = Model), show.legend = FALSE, coef = Inf)+
    scale_fill_manual(values=c("red", "orange", 
                               "blue", "skyblue",
                               "darkgreen", "yellowgreen","black"))+
    facet_wrap(~setting, scales="free_y")+
    theme_minimal()
}
dev.off()
