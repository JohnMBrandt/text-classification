setwd("/Users/johnbrandt/Documents/python_projects/nlp_final")

files = list.files("/Users/johnbrandt/Documents/python_projects/nlp_final/html/")
files <- files[grep("-EN", files)]
sdg <- read.csv("sdg-linkages.csv")

isos <- lapply(files, function(x) substr(x, 1, 3))
duplicated <- which(duplicated(isos)) - 1
duplicated[9] <- duplicated[9] + 1
duplicated[4] <- duplicated[4] + 1
duplicated[10] <- duplicated[10] + 1
files <- files[-duplicated]

proc_html <- function(i) {
  html <- readLines(paste0("html/", files[i]))
  html <- gsub("<(.|\n)*?>","",html)
  html <- html[html != ""]
  html <- html[grep(" ", html)]
  html <- unlist(strsplit(html, "[.;] "))
  html <- html[grep(" ", html)]
  html <- gsub("^ ", "", html)
  html <- gsub( " *\\[.*?\\] *", "", html)
  html <- gsub("^ ", "", html)
  periods <- which(grepl("[.]$", html))
  html[periods] <- gsub("[.]$", "", html[periods])
  html <- gsub("([a-z][.])([A-Z])", "\\1 \\2", html)
  html <- gsub( " *\\(.*?\\) *", "", html)
  html <- gsub("([0-9]{1,}),([0-9]{1,})", "\\1\\2", html)
  html <- gsub("([0-9]{1,}) ([0-9]{1,})", "\\1\\2", html)
  html <- tolower(html)
  html <- gsub("[*:]$", "", html)
  html <- gsub("'|‟|’|\\%|*", "", html)
  html <- gsub(";", "", html)
  html <- unlist(strsplit(html, "[.:] "))
  html <- gsub("^ ", "", html)
  html <- gsub("↩", "", html)
  html <- gsub("\\(|\\)", "", html)
  html <- gsub("\\*", "", html)
  html <- gsub("[.]$", "", html)
  html <- gsub("\\{|\\}|\\[|\\]", "", html)
  return(html)
}

proc_extr <- function(i) {
  ISO <- substr(files[i], 1, 3)
  afgh <- as.character(sdg$Text[sdg$ISO.code == ISO])
  afgh <- gsub( " *\\(.*?\\) *", "", afgh)
  afgh <- gsub("\\s+{2,}", " ", afgh)
  afgh <- gsub(" $", "", afgh)
  afgh <- gsub("([0-9]{1,}),([0-9]{1,})", "\\1\\2", afgh)
  afgh <- gsub("([0-9]{1,}) ([0-9]{1,})", "\\1\\2", afgh)
  afgh <- tolower(afgh)
  afgh <- gsub("[.]$", "", afgh)
  afgh <- gsub("[*:]$", "", afgh)
  afgh <- gsub("\\?", "", afgh)
  afgh <- gsub(". $", "", afgh)
  afgh <- gsub("'|’|", "", afgh)
  afgh <- gsub("\\%", "", afgh)
  afgh <- gsub("^\\s+{1,}", "", afgh)
  afgh <- gsub("\\s+{1,}$", "", afgh)
  afgh <- gsub("\\(|\\)|\\*", "", afgh)
  afgh <- gsub(" and$", "", afgh)
  afgh <- gsub("^[*><]", "", afgh)
  afgh <- gsub("^ ", "", afgh)
  afgh <- gsub(",([A-z])", ", \\1", afgh)
  afgh <- gsub("\\{|\\}|\\[|\\]", "", afgh)
  return(afgh)
}

calc_missing <- function(x) {
  html <- proc_html(x)
  afgh <- proc_extr(x)
  if(!identical(afgh, character(0))) {
    num = 0
    not_in  <- c()
    for(i in c(1:length(afgh))) {
      temp <- tolower(strsplit(afgh[i], "[.;:]")[[1]][1])
      if(sum(grepl(temp, html)) == 0) {
        num <- num + 1
        not_in <- c(not_in, temp)
      }
    }
    #print(num)
    #print(length(html))
    append_ids <- sample(length(html), num)
    for(i in seq_along(append_ids)) {
      html <- append(html, not_in[i], append_ids[i])
    }
    binary <- rep(1, length(html))
    for(i in c(1:length(html))) {
      if(sum(grepl(html[i], afgh)) == 0) {
        binary[i] <- 0
      }
    }
    write.table(as.character(binary), paste0("ndc-extraction/y/", x, ".txt"), row.names=F, quote=F, col.names = F)
    write.table(html, paste0("ndc-extraction/x/", x, ".txt"), row.names=F, quote=F, col.names = F)
  }
}

perc <- lapply(c(1:length(files)), calc_missing)


split(seq(1,74,1), rep(1:5, each = 15))
