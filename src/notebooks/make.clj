(ns notebooks.make
  (:require [scicloj.clay.v2.api :as clay]))

(def book-spec
  {:format [:quarto :html]
   :book {:title "PQ RAG Evaluation"}
   :base-target-path "book"
   :base-source-path "src"
   :source-path ["index.clj"
                 "notebooks/preparation.clj"
                 "notebooks/vector_database.clj"
                 "notebooks/vdb_evaluation.clj"
                 "notebooks/generation.clj"
                 "notebooks/rag_evaluation.clj"
                 "notebooks/single_model_eval.clj"]
   :clean-up-target-dir true})

;; TODO: test for python deps

(defn make-book [_] (clay/make! book-spec))

(comment
  (make-book nil))
