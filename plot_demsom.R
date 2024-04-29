library(ggplot2)
library(dplyr)
library(glue)


df <- arrow::read_parquet("data/results/pp_demsom_results.parquet")

df <- df %>%
  filter(preds_text != "DET") %>%
  count(maincorpus, year, labels_text)

df <- df %>%
  group_by(maincorpus, year) %>%
  mutate(total = sum(n),
         proportion = n / total)

df <- df %>%
  mutate(post_year = lubridate::ymd(year, truncated=2L),
         corpus = recode(maincorpus, gp = "GP", svt = "SVT", familjeliv = "Familjeliv", bloggmix = "Bloggmix"))

# Confidence interval bounds using normal approx of binomial: p +- z * sqrt(p(1-p)/n)
alpha <- 0.05 # 95% conf interval
df$conf_lower <- df$proportion - qnorm(p = 1 - alpha/2) * sqrt((df$proportion * (1 - df$proportion)) / df$total)
df$conf_upper <- df$proportion + qnorm(p = 1 - alpha/2) * sqrt((df$proportion * (1 - df$proportion)) / df$total)

df$conf_lower[df$conf_lower < 0] <- 0 # Set negative lower conf bound values to 0
df$conf_upper[df$conf_upper > 1] <- 1

p1 <- ggplot(data = df %>% filter(labels_text == "DE" & total > 500), 
            aes(x = post_year, y = proportion)) +
  geom_line(aes(linetype=corpus, color = corpus), linewidth = 0.6) +
  geom_point(aes(fill=corpus, shape=corpus, color = corpus), colour="black", size=1.4, stroke=0.3) +
  # geom_ribbon(aes(ymin = conf_lower, ymax = conf_upper, fill = corpus), alpha = 0.2, show.legend=FALSE) +
  theme_minimal() +
  scale_shape_manual(values = 21:25) +
  scale_x_date(breaks = scales::pretty_breaks(12),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8), breaks = scales::pretty_breaks(6),
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
       y = 'Andel "de som"',
       # title = 'Andel "de som" av totala antalet användningar av "de/dem som" efter preposition',
       color = "Korpus",
       shape = "Korpus",
       linetype = "Korpus",
       fill = "Korpus")

ggsave("plots/desom_vs_demsom/desom_vs_demsom_by_corpus.png", 
       plot = p1, width=1900, height=1000, units="px", dpi=300, bg = "white")

df %>%
  filter(labels_text == "DE" & total > 500) %>%
  rename(total_desom = n) %>%
  ungroup() %>%
  select(corpus, year, total, total_desom, proportion) %>%
  readr::write_csv("plots/desom_vs_demsom/desom_vs_demsom_by_corpus.csv")



p2 <- ggplot(data = df %>% filter(labels_text == "DE" & total > 500), 
            aes(x = post_year, y = proportion, color = corpus)) +
  geom_line(aes(linetype=corpus), linewidth = 0.8) +
  geom_point(aes(fill=corpus, shape=corpus, size=total), colour="black", stroke=0.5) +
  theme_minimal(base_size=19) +
  scale_shape_manual(values = 21:25) +
  scale_x_date(breaks = scales::pretty_breaks(12),
               guide = guide_axis(n.dodge=2)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8), breaks = scales::pretty_breaks(6),
                     expand = expansion(mult = c(0, 0), add = c(0, 0.003))) +
  scale_size_continuous(labels = function(x) format(x, big.mark = " ", decimal.mark = ".", scientific = FALSE),
                        limits = c(400, 6650),
                        breaks = c(500, 1000, 2000, 3000, 4000, 5000, 6500),
                        trans = "log",
  ) +
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
       y = 'Andel "de som"',
       # title = 'Andel "de som" av totala antalet användningar av "de/dem som" efter preposition',
       size = "Antal obs.",
       color = "Korpus",
       shape = "Korpus",
       linetype = "Korpus",
       fill = "Korpus")


ggsave("plots/desom_vs_demsom/desom_vs_demsom_by_corpus_pointsize.png", 
       plot = p2, width=1900, height=1000, units="px", dpi=150, bg = "white")



## Code to add reddit stats for "de som" vs "dem som". Not relevant anymore since we focus only on de/dem after PP.
# corpora <- dir("data/demsom_vs_desom")
# 
# df_list <- vector(mode = "list", length=length(corpora))
# 
# for (i in 1:length(corpora)){
#   df_list[[i]] <- readr::read_delim(glue("data/demsom_vs_desom/{corpora[i]}/all/all_users.tsv"))
#   names(df_list)
# }
# 
# names(df_list) <- corpora
# 
# df <- bind_rows(df_list, .id = "corpus")
# df <- df %>%
#   mutate_all(function(x) ifelse(is.nan(x), yes=NA, no=x)) %>%
#   mutate(post_year = lubridate::ymd(period, truncated=2L),
#          corpus = recode(corpus, gp = "GP", svt = "SVT", familjeliv = "Familjeliv", bloggmix = "Bloggmix"))
# 
# 
# df_reddit <- arrow::read_parquet("data/reddit/2010-04-08_2023-04-30_dedem_reddit.parquet")
# df_reddit <- df_reddit %>%
#   mutate(desom = stringr::str_locate_all(body, pattern = '(?<!\\w)[Dd][Ee] [Ss][Oo][Mm][\\.,;\\"]?(?![\\w])'),
#          demsom = stringr::str_locate_all(body, pattern = '(?<!\\w)[Dd][Ee][Mm] [Ss][Oo][Mm][\\.,;\\"]?(?![\\w])'))
# 
# df_reddit <- df_reddit %>%
#   rowwise() %>%
#   mutate(n_desom = nrow(desom),
#          n_demsom = nrow(demsom)) %>%
#   ungroup()
# 
# df_reddit_demsom <- df_reddit %>%
#   mutate(year = lubridate::year(lubridate::as_datetime(created_utc))) %>%
#   filter(!(link_id %in% c("t3_1tiyag","t3_21ivwl","t3_1nivxb","t3_1a2l0a","t3_196mcb","t3_73tfif",
#                           "t3_40xpto","t3_2t1c10","t3_2sbb4h","t3_2sa99e","t3_tstuu8","t3_trsoxk",
#                           "t3_t8ru72","t3_sjqvgi","t3_sxccgy","t3_t11trf","t3_sfhmdf","t3_rogtck",
#                           "t3_rsrn64","t3_rogtck","t3_qn59mz","t3_p9ax6t","t3_koaa9b","t3_kwuq03",
#                           "t3_koaa9b","t3_95ulyh","t3_122f5cn","t3_11tl2aq","t3_11wnn6m","t3_11sm2o3",
#                           "t3_11qd0f8","t3_11qf0p0","t3_11l2w20","t3_113mq3g","t3_zy45b3","t3_10mqsa7",
#                           "t3_zehkgi","t3_z0sdcl","t3_xjzhsv","t3_wwhlyh","t3_w1y7te","t3_ubgl37",
#                           "t3_u8k25o"))) %>%
#   group_by(year) %>%
#   summarise(total = sum(n_desom) + sum(n_demsom),
#             v1abs = sum(n_demsom),
#             v2abs = sum(n_desom)) %>%
#   rename(period = year) %>%
#   mutate(corpus = "Reddit",
#          v1rel = v1abs / total,
#          v2rel = v2abs / total,
#          post_year = lubridate::ymd(period, truncated=2L))
# 
# df <- bind_rows(df %>% select(-v1ipm, -v2ipm), df_reddit_demsom)
