
EastWestAirlines<-read.csv(file.choose())
summary(EastWestAirlines)
normalized_data<-scale(EastWestAirlines[,1:12])
d<-dist(normalized_data,method = "euclidean")#distance matrix
?dist()
fit<-hclust(d,method="complete")
str(fit)

fit$height
fit$order
fit$labels
fit$method
fit$call
fit$dist.method
plot(fit)#display dendrogram
plot(fit,hang=-1)
#how many clusters?
rect.hclust(fit, k=10,border="red")
group <-cutree(fit,k=10)
group
 ?cutree
#rect.hclust(fit,k=10,border='red')
?rect.hclust

membership <-as.matrix(group)

final <-data.frame(EastWestAirlines,membership)
final1 <-final[,c(ncol(final),1:(ncol(final)-1))]

?write.xlsx
write.csv(final1,file="final1.csv",row.names = FALSE)

getwd()
aggregate(EastWestAirlines[-1],by=list(final$membership),mean)
