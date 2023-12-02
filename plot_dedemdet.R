library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
# library(showtext)

df <- arrow::read_parquet("data/results/all_results_preprocessed.parquet")

df <- df %>%
  mutate(maincorpus = recode(maincorpus, gp = "GP", svt = "SVT", familjeliv = "Familjeliv", reddit = "Reddit", bloggmix = "Bloggmix"))

# font_add("Times New Roman", regular = "/home/faton/projects/text/sprakbanken_dedem/fonts/Times_New_Roman.ttf", 
#          bold = "/home/faton/projects/text/sprakbanken_dedem/fonts/Times_New_Roman_Bold.ttf",
#          italic = "/home/faton/projects/text/sprakbanken_dedem/fonts/Times_New_Roman_Italic.ttf",
#          bolditalic = "/home/faton/projects/text/sprakbanken_dedem/fonts/Times_New_Roman_Bold_Italic.ttf")
# showtext_auto()

preprocess_maincorpus_data <- function(df, mistake = NULL, score_threshold = 0, group_var = maincorpus, alpha=0.05, author_level=FALSE){
  group_var <- enquo(group_var)
  
  if (!is.null(mistake)){
    if (mistake == "de_mistake"){
      df <- df %>% 
        filter(scores_only > score_threshold) %>%
        filter((entity == "dem" & word == "dem") | (entity == "dem" & word == "de")) %>%
        mutate(ratio = de_mistake)
    } else if (mistake == "dem_mistake"){
      df <- df %>% 
        filter(scores_only > score_threshold) %>%
        filter((entity == "de" & word == "de") | (entity == "de" & word == "dem")) %>%
        mutate(ratio = dem_mistake)
    } else if (mistake == "det_mistake"){
      df <- df %>% 
        filter(scores_only > score_threshold) %>%
        filter((entity == "de" & word == "de") | (entity == "det" & word == "de")) %>%
        mutate(ratio = det_mistake)
    } else if (mistake == "de_mistake_threshold"){
      df <- df %>%
        filter((entity_threshold == "dem" & word == "dem") | (entity_threshold == "dem" & word == "de")) %>%
        mutate(ratio = de_mistake_threshold)
    } else if (mistake == "dem_mistake_threshold"){
      df <- df %>%
        filter((entity_threshold == "de" & word == "de") | (entity_threshold == "de" & word == "dem")) %>%
        mutate(ratio = dem_mistake_threshold)
    } else if (mistake == "det_mistake_threshold"){
      df <- df %>%
        filter((entity_threshold == "de" & word == "de") | (entity_threshold == "det" & word == "de")) %>%
        mutate(ratio = det_mistake_threshold)
    }
  }
  print(df)
  
  df_gen <- df %>%
    group_by(post_year, !!group_var) %>%
    summarise(mean_ratio = mean(ratio), 
              n_obs = n(),
              n_authors = length(unique(speaker)),
              max_author_comments = max(author_comments_per_year),
              max_de_mistake = max(author_de_mistake),
              max_dem_mistake = max(author_dem_mistake),
              max_det_mistake = max(author_det_mistake),
              total_de_mistake = sum(de_mistake),
              total_dem_mistake = sum(dem_mistake),
              total_det_mistake = sum(det_mistake)) %>%
    ungroup()
  
  # Confidence interval bounds using normal approx of binomial: p +- z * sqrt(p(1-p)/n)
  df_gen$conf_lower <- df_gen$mean_ratio - qnorm(p = 1 - alpha/2) * sqrt((df_gen$mean_ratio * (1 - df_gen$mean_ratio)) / df_gen$n_obs)
  df_gen$conf_upper <- df_gen$mean_ratio + qnorm(p = 1 - alpha/2) * sqrt((df_gen$mean_ratio * (1 - df_gen$mean_ratio)) / df_gen$n_obs)
  
  df_gen$conf_lower[df_gen$conf_lower < 0] <- 0 # Set negative lower conf bound values to 0
  df_gen$conf_upper[df_gen$conf_upper > 1] <- 1
  
  if (author_level == TRUE){
    # Analyze mistake ratios on author level instead of corpus/generation level
    df_gen <- df %>%
      group_by(post_year, !!group_var, speaker) %>%
      mutate(mean_ratio_author = mean(ratio)) %>%
      group_by(post_year, !!group_var) %>%
      summarise(n_obs = n(),
                n_authors = length(unique(speaker)),
                max_author_comments = max(author_comments_per_year),
                max_de_mistake = max(author_de_mistake),
                max_dem_mistake = max(author_dem_mistake),
                max_det_mistake = max(author_det_mistake),
                total_de_mistake = sum(de_mistake),
                total_dem_mistake = sum(dem_mistake),
                total_det_mistake = sum(det_mistake)) %>%
      ungroup()
    
    df_gen_mean <- df %>%
      group_by(year) %>%
      add_count(speaker, name="author_comments_per_year") %>%
      group_by(post_year, !!group_var, speaker) %>%
      mutate(mean_ratio_author = mean(ratio)) %>%
      summarise(mean_ratio = first(mean_ratio_author),
                max_author_comments =  max(author_comments_per_year)) %>%
      # filter(max_author_comments >= 10) %>%
      group_by(post_year, !!group_var) %>%
      summarise(mean_ratio = mean(mean_ratio))
    
    df_gen <- df_gen %>%
      left_join(df_gen_mean, by = c("post_year" = "post_year", "generation" = "generation"))
    
    
  }
  
  return(df_gen)
}

plot_trend_maincorpus <- function(df_gen, title, group_var=maincorpus, legend_title="Korpus", y_max_limit=NA, 
                                  scale_pointsize=FALSE, confidence_interval=FALSE){
  group_var <- enquo(group_var)
  
  if (scale_pointsize == FALSE){
    p <- ggplot(data=df_gen, aes(x = post_year, y=mean_ratio)) +
      geom_line(aes(linetype=!!group_var, colour=!!group_var), linewidth = 0.6) +
      # geom_point(aes(shape=maincorpus), size=2.1, color="black") +
      geom_point(aes(fill=!!group_var, shape=!!group_var, colour=!!group_var), colour="black", size=1.4, stroke=0.3) +
      theme_minimal() +
      scale_shape_manual(values = 21:25) +
      scale_x_date(breaks = scales::pretty_breaks(12),
                   guide = guide_axis(n.dodge=2)) +
      scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, y_max_limit), breaks = scales::pretty_breaks(6),
                         expand = expansion(mult = c(0, 0), add = c(0, 0.003))) +
      theme(plot.title = element_text(size=11),
            plot.margin = margin(t=0.3, r=0.15, b=0.1, l=0.15, unit = "cm"),
            plot.background = element_rect(colour="black", linewidth = 0.05),
            axis.line = element_line(colour = "black", linewidth = 0.3, linetype = 1),
            axis.ticks = element_line(linewidth = 0.1),
            axis.text.x = element_text(vjust=0.1, color = "black"),
            axis.text.y = element_text(color = "black"),
            panel.grid.major.y = element_line(colour = "black", linewidth = 0.1, linetype = 1),
            panel.grid.minor.y = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank(),
            text=element_text(family="Times New Roman", size=12),
            plot.subtitle=element_text(size=5.5),
            legend.key.width = unit(1.0, "cm")) +
      guides(linetype = guide_legend(override.aes = list(size = 1.5))) +
      labs(x = "År",
           y = "Förväxlingar",
           # title = title,
           color = legend_title,
           shape = legend_title,
           linetype = legend_title,
           fill = legend_title)
  } else {
    # Scale the geom_point size by number of observations per data point
    p <- ggplot(data=df_gen, aes(x = post_year, y=mean_ratio)) +
      geom_line(aes(linetype=!!group_var, colour=!!group_var), linewidth = 1) +
      # geom_point(aes(shape=maincorpus), size=2.1, color="black") +
      geom_point(aes(fill=!!group_var, shape=!!group_var, size=n_obs, colour=!!group_var), colour="black", stroke=0.5) +
      theme_minimal() +
      scale_shape_manual(values = 21:25) +
      scale_x_date(breaks = scales::pretty_breaks(12),
                   guide = guide_axis(n.dodge=2)) +
      scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, y_max_limit), breaks = scales::pretty_breaks(6),
                         expand = expansion(mult = c(0, 0), add = c(0, 0.005))) +
      theme(plot.title = element_text(size=11),
            plot.margin = margin(t=0.3, r=0.15, b=0.1, l=0.15, unit = "cm"),
            plot.background = element_rect(colour="black", linewidth = 0.05),
            axis.line = element_line(colour = "black", linewidth = 0.3, linetype = 1),
            axis.ticks = element_line(linewidth = 0.1),
            axis.text.x = element_text(vjust=0.1, color = "black"),
            axis.text.y = element_text(color = "black"),
            panel.grid.major.y = element_line(colour = "black", linewidth = 0.1, linetype = 1),
            panel.grid.minor.y = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank(),
            text=element_text(family="Times New Roman", size=24),
            plot.subtitle=element_text(size=5.5),
            legend.key.width = unit(1.0, "cm"),
            legend.key.height = unit(0.8, "cm")) +
      guides(linetype = guide_legend(override.aes = list(size = 2.5), order=1),
             shape = guide_legend(order=1),
             fill = guide_legend(order=1),
             color = guide_legend(order=1),
             size = guide_legend(order=2, override.aes = list(color = "grey15"))) +
      labs(x = "År",
           y = "Förväxlingar",
           # title = title,
           size = "Antal obs.",
           color = legend_title,
           shape = legend_title,
           linetype = legend_title,
           fill = legend_title)
  }
  
  if (confidence_interval == TRUE){
    p <- p + geom_ribbon(aes(ymin = conf_lower, ymax = conf_upper, fill = !!group_var), alpha = 0.25, show.legend=FALSE)
  }
  
  return(p)
}


#### Corpus Level ####
df_gen <- preprocess_maincorpus_data(df = df %>% filter(!(maincorpus == "Familjeliv" & yob == 1970)), mistake = "de_mistake", score_threshold = 0)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_de_mistake/total_de_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3 & !is.na(max_author_comments), yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500)

p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt', confidence_interval=FALSE)
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_satsdel_funktion/corpus/de_objektsposition_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")

# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt', scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(600, 60000),
                               breaks = c(700, 2000, 4000, 6000, 10000, 20000, 60000),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/corpus/pointsize/de_objektsposition_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")



df_gen <- preprocess_maincorpus_data(df = df %>% filter(!(maincorpus == "Familjeliv" & yob == 1970)), mistake = "dem_mistake", score_threshold = 0)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_dem_mistake/total_dem_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3 & !is.na(max_author_comments), yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare', confidence_interval = FALSE)
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_satsdel_funktion/corpus/dem_subjektdeterminerare_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")


# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare', scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(700, 340000),
                               breaks = c(800, 5000, 10000, 50000, 100000, 200000, 340000),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/corpus/pointsize/dem_subjektdeterminerare_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")



df_gen <- preprocess_maincorpus_data(df = df %>% filter(!(maincorpus == "Familjeliv" & yob == 1970)), mistake = "det_mistake", score_threshold = 0)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_det_mistake/total_det_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3 & !is.na(max_author_comments), yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular)', confidence_interval = TRUE)
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_form/corpus/det_by_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")

# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare', scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(700, 405000),
                               breaks = c(800, 5000, 10000, 50000, 100000, 200000, 400000),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/corpus/pointsize/det_by_maincorpus.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")



##### By generation (ratio of mistakes/(total usage) for all comments in generation)
df_familjeliv <- df %>% 
  filter(maincorpus == "Familjeliv") %>%
  filter(yob != 1970)

df_gen <- preprocess_maincorpus_data(df = df_familjeliv, mistake = "de_mistake", score_threshold = 0, group_var = generation)
df_gen <- df_gen %>%
  filter(n_obs > 500 & n_authors > 20) 
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt över generationer', group_var = generation, 
                           legend_title = "Generation", y_max_limit = 0.108, confidence_interval = FALSE)
ggsave("plots/analys_satsdel_funktion/generation/de_objektsposition_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")

# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt över generationer', 
                           group_var = generation, legend_title = "Generation", y_max_limit = 0.108, scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(420, 20000),
                               breaks = c(500, 1000, 2000, 3000, 5000, 10000, 19600),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/generation/pointsize/de_objektsposition_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")


# Dem as subject/determiner
df_gen <- preprocess_maincorpus_data(df = df_familjeliv, mistake = "dem_mistake", score_threshold = 0, group_var = generation)
df_gen <- df_gen %>%
  filter(n_obs > 500 & n_authors > 20)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare över generationer', group_var = generation, 
                           legend_title = "Generation", y_max_limit = 0.32, confidence_interval = FALSE)
ggsave("plots/analys_satsdel_funktion/generation/dem_subjektdeterminerare_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")

# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare över generationer', 
                           group_var = generation, legend_title = "Generation", y_max_limit = 0.31, scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(600, 175000),
                               breaks = c(700, 2000, 5000, 10000, 20000, 50000, 109000),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/generation/pointsize/dem_subjektdeterminerare_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")


df_gen <- preprocess_maincorpus_data(df = df_familjeliv, mistake = "det_mistake", score_threshold = 0, group_var = generation)
df_gen <- df_gen %>%
  filter(n_obs > 500 & n_authors > 20)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular) över generationer', 
                           group_var = generation, legend_title = "Generation")
ggsave("plots/analys_form/generation/det_by_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")

# Another version with pointsize scaled
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular) över generationer', 
                           group_var = generation, legend_title = "Generation", scale_pointsize = TRUE)
p <- p + scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                               limits = c(900, 205000),
                               breaks = c(1000, 2500, 5000, 10000, 20000, 50000, 167000),
                               trans = "log") 
ggsave("plots/analys_satsdel_funktion/generation/pointsize/dem_subjektdeterminerare_familjeliv_generation.png", 
       plot = p, width=1900, height=1000, units="px", dpi=150, bg = "white")



#### By Generation -- Author Level (mean of authors' individual mistake ratios) ####
df_gen <- preprocess_maincorpus_data(df = df_familjeliv %>% filter(author_comments_per_year >= 10), 
                                     mistake = "de_mistake", score_threshold = 0, group_var = generation, author_level = TRUE)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_de_mistake/total_de_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3, yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500 & n_authors > 20)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt över generationer', group_var = generation, legend_title = "Generation", y_max_limit = 0.24)
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_satsdel_funktion/generation/author/de_objektsposition_familjeliv_generation_author.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")


df_gen <- preprocess_maincorpus_data(df = df_familjeliv %>% filter(author_comments_per_year >= 10), 
                                     mistake = "dem_mistake", score_threshold = 0, group_var = generation, author_level = TRUE)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_dem_mistake/total_dem_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3, yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500 & n_authors > 20)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare över generationer', group_var = generation, legend_title = "Generation", y_max_limit = 0.146)
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_satsdel_funktion/generation/author/dem_subjektdeterminerare_familjeliv_generation_author.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")


df_gen <- preprocess_maincorpus_data(df = df_familjeliv %>% filter(author_comments_per_year >= 10),
                                     mistake = "det_mistake", score_threshold = 0, group_var = generation, author_level=TRUE)
df_gen <- df_gen %>%
  mutate(single_author_ratio = max_det_mistake/total_det_mistake,
         single_author_over_50 = ifelse(single_author_ratio > 0.3, yes = mean_ratio, no = NA)) %>%
  filter(n_obs > 500 & n_authors > 20)
p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular) över generationer', group_var = generation, legend_title = "Generation")
# p <- p + geom_point(aes(y = df_gen$single_author_over_50), shape=4, colour="black", size=2, alpha=0.85)
ggsave("plots/analys_form/generation/author/det_by_familjeliv_generation_author.png", 
       plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")



# a <- reddit %>%
#   filter(stringr::str_detect(sentence, "(?<!\\w)[Dd][Ee] och [Dd][Ee][Mm]?(?![\\w])")) %>%
#   select(permalink)



# df_gen <- df_gen %>%
#   select(post_year, generation, n_authors, n_obs, total_de_mistake, max_de_mistake, mean_ratio) %>%
#   mutate(single_author_ratio = max_de_mistake/total_de_mistake) %>%
#   rename(mistake_ratio = mean_ratio) %>%
#   arrange(post_year, generation)
# clipr::write_clip(df_gen)


# df_gen <- preprocess_maincorpus_data(df = df, mistake = "det_mistake", score_threshold = 0.95)
# p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular)')
# ggsave("plots/analys_form/score_threshold_95_filter_remove/det_by_maincorpus.png", 
#        plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")
# 
# df_gen <- preprocess_maincorpus_data(df = df, mistake = "det_mistake_threshold", score_threshold = 0.95)
# p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular)')
# ggsave("plots/analys_form/score_threshold_95_keep_all/det_by_maincorpus.png", 
#        plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")



# ##### corpus_type group var ####
# df_gen <- preprocess_maincorpus_data(df = df, mistake = "de_mistake", score_threshold = 0, group_var = corpus_type)
# df_gen <- df_gen %>%
#   filter(n_obs > 500)
# p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "de" som objekt efter medietyp',
#                            group_var = corpus_type, legend_title = "Medietyp")
# ggsave("plots/analys_satsdel_funktion/score_threshold_0/de_objektsposition_mediatype.png",
#        plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")
# 
# 
# df_gen <- preprocess_maincorpus_data(df = df, mistake = "dem_mistake", score_threshold = 0, group_var = corpus_type)
# df_gen <- df_gen %>%
#   filter(n_obs > 500)
# p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel användningar av "dem" som subjekt och determinerare efter medietyp',
#                            group_var = corpus_type, legend_title = "Medietyp")
# ggsave("plots/analys_satsdel_funktion/score_threshold_0/dem_subjektdeterminerare_mediatype.png",
#        plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")
# 
# 
# df_gen <- preprocess_maincorpus_data(df = df, mistake = "det_mistake", score_threshold = 0, group_var = corpus_type)
# p <- plot_trend_maincorpus(df_gen = df_gen, title='Andel "de" som borde vara "det" (singular) efter medietyp',
#                            group_var = corpus_type, legend_title = "Medietyp")
# ggsave("plots/analys_form/score_threshold_0/det_by_mediatype.png",
#        plot = p, width=1900, height=1000, units="px", dpi=300, bg = "white")
