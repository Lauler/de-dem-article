library(readr)
library(dplyr)
library(tidyr)

df <- dplyr::bind_rows(arrow::read_parquet("data/results/gp_results.parquet"),
                       arrow::read_parquet("data/results/svt_results.parquet"),
                       arrow::read_parquet("data/results/bloggmix_results.parquet"),
                       arrow::read_parquet("data/results/familjeliv_age_results.parquet"))


reddit <- arrow::read_parquet("data/results/reddit_results.parquet")
reddit <- reddit %>%
  rename(unique_id = id, speaker = author) %>%
  mutate(token_id = token_index, subcorpus = maincorpus) %>%
  filter(!(link_id %in% c("t3_1tiyag","t3_21ivwl","t3_1nivxb","t3_1a2l0a","t3_196mcb","t3_73tfif",
                          "t3_40xpto","t3_2t1c10","t3_2sbb4h","t3_2sa99e","t3_tstuu8","t3_trsoxk",
                          "t3_t8ru72","t3_sjqvgi","t3_sxccgy","t3_t11trf","t3_sfhmdf","t3_rogtck",
                          "t3_rsrn64","t3_rogtck","t3_qn59mz","t3_p9ax6t","t3_koaa9b","t3_kwuq03",
                          "t3_koaa9b","t3_95ulyh","t3_122f5cn","t3_11tl2aq","t3_11wnn6m","t3_11sm2o3",
                          "t3_11qd0f8","t3_11qf0p0","t3_11l2w20","t3_113mq3g","t3_zy45b3","t3_10mqsa7",
                          "t3_zehkgi","t3_z0sdcl","t3_xjzhsv","t3_wwhlyh","t3_w1y7te","t3_ubgl37",
                          "t3_u8k25o"))) %>% # Threads about de/dem/dom-debate.
  filter(!stringr::str_detect(sentence, "(?<!\\w)[Dd][Ee] och [Dd][Ee][Mm]?(?![\\w])")) %>%
  filter(!stringr::str_detect(sentence, "(?<!\\w)[Dd][Ee]/[Dd][Ee][Mm]?(?![\\w])")) %>%
  select(unique_id, token_id, sentence, maincorpus, subcorpus, year, sentence_length, 
         begin_char, end_char, token_index, scores_only, labels_text, preds_text, is_relative_clause, speaker, date)

df <- df %>%
  select(-preds_only, -labels_only) %>%
  mutate(unique_id = as.character(unique_id))

df <- bind_rows(df, reddit)

df <- df %>%
  mutate(entity = stringr::str_to_lower(preds_text),
         word = stringr::str_to_lower(labels_text),
         post_year = lubridate::ymd(year, truncated=2L),
         generation = recode(generation, Gen1 = "Före 1970", Gen2 = "1970-1979", Gen3 = "1980-1989", Gen4 = "Från 1990"),
         generation = factor(generation, levels = c("Från 1990", "1980-1989", "1970-1979", "Före 1970")))


# We use our predicted entities regardless of score for {word}_mistake
#
# We also set a threshold and adjust predictions based on the score threshold.
# If score > threshold, we keep the model prediction, otherwise we keep the word
# that was originally written in the text.
df <- df %>%
  filter(is_relative_clause == FALSE) %>%
  mutate(de_mistake = ifelse(entity == "dem" & word == "de", yes = TRUE, no = FALSE),
         dem_mistake = ifelse(entity == "de" & word == "dem", yes = TRUE, no = FALSE),
         det_mistake = ifelse(entity == "det" & word == "de", yes = TRUE, no = FALSE),
         all_mistake = de_mistake | dem_mistake | det_mistake,
         entity_threshold = ifelse(scores_only > 0.95, yes = entity, no = word),
         de_mistake_threshold = ifelse(entity_threshold == "dem" & word == "de", yes = TRUE, no = FALSE),
         dem_mistake_threshold = ifelse(entity_threshold == "de" & word == "dem", yes = TRUE, no = FALSE),
         det_mistake_threshold = ifelse(entity_threshold == "det" & word == "de", yes = TRUE, no = FALSE),
         corpus_type = ifelse(maincorpus == "svt" | maincorpus == "gp", yes = "Nyheter", no = "Sociala medier"))


df <- df %>% 
  add_count(speaker, name="author_comments_total") %>%
  group_by(year) %>%
  add_count(speaker, name="author_comments_per_year") %>%
  mutate(author_comments_per_year = ifelse(is.na(speaker), yes = NA, no = author_comments_per_year)) %>%
  ungroup()


df <- df %>%
  group_by(year, speaker) %>%
  mutate(author_de_mistake = sum(de_mistake),
         author_dem_mistake = sum(dem_mistake),
         author_det_mistake = sum(det_mistake),
         author_de_correct = sum(de_mistake == FALSE & det_mistake == FALSE & word == "de"),
         author_dem_correct = sum(dem_mistake == FALSE & word == "dem"),
         author_all_mistake = sum(all_mistake),
         author_all_mistake_ratio = author_all_mistake/author_comments_per_year) %>%
  ungroup()

arrow::write_parquet(df, "data/results/all_results_preprocessed.parquet")
