(define (domain ferry)
  (:requirements :strips :typing)
  (:types car location)
  (:predicates
    (not-eq ?x - location ?y - location)
    (at-ferry ?l - location)
    (at ?c - car ?l - location)
    (on ?c - car)
    (empty-ferry)
  )

  (:action sail
    :parameters (?from - location ?to - location)
    :precondition (and (not-eq ?from ?to) (at-ferry ?from))
    :effect (and (not (at-ferry ?from)) (at-ferry ?to))
  )

  (:action board
    :parameters (?car - car ?loc - location)
    :precondition (and (at ?car ?loc) (at-ferry ?loc) (empty-ferry))
    :effect (and (on ?car) (not (at ?car ?loc)) (not (empty-ferry)))
  )

  (:action debark
    :parameters (?car - car ?loc - location)
    :precondition (and (on ?car) (at-ferry ?loc))
    :effect (and (not (on ?car)) (at ?car ?loc) (empty-ferry))
  )
)
