remove_outliers <- function(data,sigma){
  
  numcols <- which(sapply(data,is.numeric) == T)
  means <- apply(data%>%select(numcols),2,mean)
  sdv <- apply(data%>%select(numcols),2,sd)
  print(sdv)
  rows <- NULL
  for(i in 1:length(numcols)){
    index <- which(data%>%select(numcols[i]) >= (means[i]+ (sigma*sdv[i])) |  data%>%select(numcols[i]) <= (means[i] - (sigma*sdv[i])))
    rows <- append(rows,index)
    
  }
  rows <- unique(rows)
  data[-rows,]
  
}



