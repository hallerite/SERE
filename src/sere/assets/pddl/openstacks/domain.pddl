;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Open-stacks domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain openstacks)
  (:requirements :strips :typing :adl :action-costs)
  (:types robot order product)
  (:predicates (open ?o - order)
               (shipped ?o - order)
               (requires ?o - order ?p - product)
               (made ?p - product))
  (:functions (open-count) - number
              (stack-limit) - number
              (remaining ?o - order) - number)

  (:action open-order
    :parameters (?r - robot ?o - order)
    :precondition (and (not (open ?o))
                       (not (shipped ?o))
                       (< (open-count) (stack-limit))
                       (forall (?p - product)
                         (or (not (requires ?o ?p))
                             (not (made ?p)))))
    :effect (and (open ?o)
                 (increase (open-count) 1)))

  (:action make-product
    :parameters (?r - robot ?p - product)
    :precondition (and (not (made ?p))
                       (forall (?o - order)
                         (or (not (requires ?o ?p))
                             (open ?o)
                             (shipped ?o))))
    :effect (and (made ?p)
                 (forall (?o - order)
                   (when (and (requires ?o ?p)
                              (open ?o)
                              (= (remaining ?o) 1))
                     (and (shipped ?o)
                          (not (open ?o))
                          (decrease (open-count) 1)
                          (decrease (remaining ?o) 1))))
                 (forall (?o - order)
                   (when (and (requires ?o ?p)
                              (open ?o)
                              (> (remaining ?o) 1))
                     (decrease (remaining ?o) 1)))))
)
