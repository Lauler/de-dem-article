library(dplyr)
library(tidyr)
library(ggplot2)

reddit <- arrow::read_parquet("data/results/reddit_results.parquet")

reddit %>%
  mutate(word_count = stringr::str_count(sentence)) %>%
  summarize(sum_word = sum(word_count))

reddit <- reddit %>%
  rename(unique_id = id) %>%
  mutate(token_id = token_index, subcorpus = maincorpus) %>%
  filter(!(link_id %in% c("t3_1tiyag","t3_21ivwl","t3_1nivxb","t3_1a2l0a","t3_196mcb","t3_73tfif",
                          "t3_40xpto","t3_2t1c10","t3_2sbb4h","t3_2sa99e","t3_tstuu8","t3_trsoxk",
                          "t3_t8ru72","t3_sjqvgi","t3_sxccgy","t3_t11trf","t3_sfhmdf","t3_rogtck",
                          "t3_rsrn64","t3_rogtck","t3_qn59mz","t3_p9ax6t","t3_koaa9b","t3_kwuq03",
                          "t3_koaa9b","t3_95ulyh","t3_122f5cn","t3_11tl2aq","t3_11wnn6m","t3_11sm2o3",
                          "t3_11qd0f8","t3_11qf0p0","t3_11l2w20","t3_113mq3g","t3_zy45b3","t3_10mqsa7",
                          "t3_zehkgi","t3_z0sdcl","t3_xjzhsv","t3_wwhlyh","t3_w1y7te","t3_ubgl37",
                          "t3_u8k25o"))) %>%
  filter(!stringr::str_detect(sentence, "(?<!\\w)[Dd][Ee] och [Dd][Ee][Mm]?(?![\\w])")) %>%
  filter(!stringr::str_detect(sentence, "(?<!\\w)[Dd][Ee]/[Dd][Ee][Mm]?(?![\\w])"))
  
reddit_karma <- reddit %>%
  filter(is_relative_clause == FALSE) %>%
  mutate(entity = stringr::str_to_lower(preds_text),
         word = stringr::str_to_lower(labels_text),
         post_year = lubridate::ymd(year, truncated=2L),
         de_mistake = ifelse(entity == "dem" & word == "de", yes = score, no = NA),
         dem_mistake = ifelse(entity == "de" & word == "dem", yes = score, no = NA),
         det_mistake = ifelse(entity == "det" & word == "de", yes = score, no = NA),
         all_mistake = is.na(de_mistake) | is.na(dem_mistake) | is.na(det_mistake),
         all_mistake = ifelse(all_mistake == TRUE, yes = score, no = NA),
         year = lubridate::ymd(year, truncated=2L)) %>%
  pivot_longer(cols = c(de_mistake, dem_mistake, det_mistake, all_mistake),
               names_to = "mistake_type",
               values_to = "ratio") %>%
  group_by(year, mistake_type) %>%
  summarize(mean_score = mean(ratio, na.rm=TRUE))

reddit_karma <- reddit_karma %>%
  group_by(year) %>%
  mutate(score_norm = mean_score/mean_score[mistake_type == "all_mistake"]) %>%
  filter(mistake_type != "det_mistake")

p1 <- ggplot(reddit_karma, aes(x = year, y = score_norm, color=mistake_type)) +
  geom_line(aes(linetype=mistake_type), linewidth = 0.6) +
  geom_point(aes(fill=mistake_type, shape=mistake_type), colour="black", size=1.4, stroke=0.3) +
  theme_minimal(base_size=9) +
  scale_color_hue(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)")) +
  scale_shape_manual(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)"), values = 21:25)  +
  scale_linetype_manual(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)"), values = 1:4) +
  scale_fill_hue(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)")) + 
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0.5, NA), breaks = scales::pretty_breaks(6)) +
  theme(plot.title = element_text(size=9),
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
        legend.key.width = unit(1.0, "cm"),
        legend.key = element_rect(color = NA, fill = NA),
        legend.key.size = unit(0.7, "cm")) +
  labs(x = "År",
       y = "Relativ karma (procent)",
       # title = "Andel karma per inlägg på Reddit för olika förväxlingar i relation till skriftspråksnorm (röd linje)",
       color = "Typ av användning",
       shape = "Typ av användning",
       linetype = "Typ av användning",
       fill = "Typ av användning")



p2 <- ggplot(reddit_karma, aes(x = year, y = mean_score, color=mistake_type)) +
  geom_line(aes(linetype=mistake_type)) +
  geom_point(aes(fill=mistake_type, shape=mistake_type), colour="black", size=1.4, stroke=0.3) +
  theme_minimal(base_size=9) +
  scale_color_hue(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)")) +
  scale_shape_manual(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)"), values = 21:25)  +
  scale_linetype_manual(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)"), values = 1:4) +
  scale_fill_hue(labels = c("Normenligt de/dem", "Normbrytande de \n(istället för dem)", "Normbrytande dem \n(istället för de)")) + 
  scale_x_date(breaks = scales::pretty_breaks(16),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(limits = c(0.5, NA), breaks = scales::pretty_breaks(6)) +
  theme(plot.title = element_text(size=10),
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
        legend.key.width = unit(1.0, "cm"),
        legend.key = element_rect(color = NA, fill = NA),
        legend.key.size = unit(0.7, "cm")) +
  labs(x = "År",
       y = "Karmapoäng (medel)",
       # title = "Karmapoäng per inlägg på Reddit uppdelat efter typ av förväxling",
       color = "Typ av användning",
       shape = "Typ av användning",
       linetype = "Typ av användning",
       fill = "Typ av användning")


ggsave("plots/reddit/karma_relative_by_mistaketype.png", 
       plot = p1, width=1900, height=1000, units="px", dpi=300, bg = "white")


ggsave("plots/reddit/karma_points_by_mistaketype.png", 
       plot = p2, width=1900, height=1000, units="px", dpi=300, bg = "white")

