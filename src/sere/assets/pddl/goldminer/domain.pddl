;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Gold-miner domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain goldminer)
  (:requirements :strips :typing)
  (:types loc)
  (:predicates (robot-at ?x - loc)
               (bomb-at ?x - loc)
               (laser-at ?x - loc)
               (soft-rock-at ?x - loc)
               (hard-rock-at ?x - loc)
               (gold-at ?x - loc)
               (arm-empty)
               (holds-bomb)
               (holds-laser)
               (holds-gold)
               (clear ?x - loc)
               (connected ?x - loc ?y - loc))

  (:action move
    :parameters (?x - loc ?y - loc)
    :precondition (and (robot-at ?x)
                       (connected ?x ?y)
                       (clear ?y))
    :effect (and (not (robot-at ?x))
                 (robot-at ?y)))

  (:action pickup-laser
    :parameters (?x - loc)
    :precondition (and (robot-at ?x)
                       (laser-at ?x)
                       (arm-empty))
    :effect (and (not (arm-empty))
                 (not (laser-at ?x))
                 (holds-laser)))

  (:action pickup-bomb
    :parameters (?x - loc)
    :precondition (and (robot-at ?x)
                       (bomb-at ?x)
                       (arm-empty))
    :effect (and (not (arm-empty))
                 (holds-bomb)))

  (:action putdown-laser
    :parameters (?x - loc)
    :precondition (and (robot-at ?x)
                       (holds-laser))
    :effect (and (not (holds-laser))
                 (arm-empty)
                 (laser-at ?x)))

  (:action detonate-bomb
    :parameters (?x - loc ?y - loc)
    :precondition (and (robot-at ?x)
                       (holds-bomb)
                       (connected ?x ?y)
                       (soft-rock-at ?y))
    :effect (and (not (holds-bomb))
                 (not (soft-rock-at ?y))
                 (arm-empty)
                 (clear ?y)))

  (:action fire-laser
    :parameters (?x - loc ?y - loc)
    :precondition (and (robot-at ?x)
                       (holds-laser)
                       (connected ?x ?y))
    :effect (and (not (soft-rock-at ?y))
                 (not (gold-at ?y))
                 (not (hard-rock-at ?y))
                 (clear ?y)))

  (:action pick-gold
    :parameters (?x - loc)
    :precondition (and (robot-at ?x)
                       (arm-empty)
                       (gold-at ?x))
    :effect (and (not (arm-empty))
                 (holds-gold)))
)
