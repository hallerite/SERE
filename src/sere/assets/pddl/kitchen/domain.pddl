(define (domain kitchen)
  (:requirements :strips :typing :fluents :derived-predicates
                 :negative-preconditions :disjunctive-preconditions :equality)

  ;; ========== Types ==========
  (:types
    robot location object number
    movable appliance - object
    container - movable
    stove - appliance
    cookware - container
    food - movable
  )

  ;; ========== Predicates ==========
  (:predicates
    (at ?r - robot ?l - location)
    (obj-at ?o - object ?l - location)
    (holding ?r - robot ?o - object)
    (clear-hand ?r - robot)
    (in ?o - object ?c - container)
    (open ?c - container)
    (powered ?a - appliance)
    (burner-on ?s - stove)
    (on-stove ?c - cookware ?s - stove)
    (has-water ?c - container)
    (has-hot-water ?m - container)
    (tea-ready ?m - container)
    (spilled ?m - container)
    (has-charger ?l - location)
    (adjacent ?l1 - location ?l2 - location)
    (openable ?c - container)
    (is-sink ?s - container)
    (pour-out-needs-open ?c - container)
    (pour-out-needs-closed ?c - container)
    (pour-in-needs-open ?c - container)
    ;; derived predicates declared here for typing
    (co-located ?x - object ?y - object)
    (boiled ?a - appliance)
    (hot-surface ?s - stove)
    (cooked ?f - food)
  )

  ;; ========== Functions / Fluents ==========
  (:functions
    (energy ?r - robot)
    (battery-cap ?r - robot)
    (water-temp ?c - container)
    (surface-temp ?s - stove)
    (food-temp ?f - food)
  )

  ;; ========== Derived Predicates ==========
  ;; co-located: robot-at / robot-at
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (at ?x ?l) (at ?y ?l))))

  ;; co-located: robot-at / obj-at
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (at ?x ?l) (obj-at ?y ?l))))

  ;; co-located: obj-at / robot-at
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (obj-at ?x ?l) (at ?y ?l))))

  ;; co-located: obj-at / obj-at
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (obj-at ?x ?l) (obj-at ?y ?l))))

  ;; co-located: x in container that is co-located with y
  (:derived (co-located ?x - object ?y - object)
    (exists (?c - container)
      (and (in ?x ?c) (co-located ?c ?y))))

  ;; co-located: y in container that is co-located with x
  (:derived (co-located ?x - object ?y - object)
    (exists (?c - container)
      (and (in ?y ?c) (co-located ?x ?c))))

  ;; co-located: robot holding x, robot co-located with y
  (:derived (co-located ?x - object ?y - object)
    (exists (?r - robot)
      (and (holding ?r ?x) (co-located ?r ?y))))

  ;; co-located: robot holding y, x co-located with robot
  (:derived (co-located ?x - object ?y - object)
    (exists (?r - robot)
      (and (holding ?r ?y) (co-located ?x ?r))))

  ;; boiled: water temp >= 100
  (:derived (boiled ?a - appliance)
    (and (has-water ?a) (>= (water-temp ?a) 100)))

  ;; hot-surface: surface temp >= 120
  (:derived (hot-surface ?s - stove)
    (>= (surface-temp ?s) 120))

  ;; cooked: food temp >= 70
  (:derived (cooked ?f - food)
    (>= (food-temp ?f) 70))

  ;; ========== Actions ==========

  ;; --- wait ---
  ;; Success: no state change (noop)
  (:action wait
    :parameters (?n - number)
    :precondition (and)
    :effect (and)
  )

  ;; --- move ---
  ;; Success: (at ?r ?to), delete (at ?r ?from), (decrease (energy ?r) 1)
  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
      (adjacent ?from ?to)
      (>= (energy ?r) 1)
      (not (= ?from ?to))
    )
    :effect (and)
  )

  ;; --- open ---
  ;; Success: (open ?c), (decrease (energy ?r) 0.2)
  (:action open
    :parameters (?r - robot ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (openable ?c)
      (not (open ?c))
    )
    :effect (and)
  )

  ;; --- close ---
  ;; Success: delete (open ?c), (decrease (energy ?r) 0.2)
  (:action close
    :parameters (?r - robot ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (openable ?c)
      (open ?c)
    )
    :effect (and)
  )

  ;; --- pick-up ---
  ;; Success: (holding ?r ?o), delete (obj-at ?o ?l) (clear-hand ?r) (on-stove ?o ?s),
  ;;          (decrease (energy ?r) 0.3)
  (:action pick-up
    :parameters (?r - robot ?o - movable)
    :precondition (and
      (co-located ?r ?o)
      (clear-hand ?r)
    )
    :effect (and)
  )

  ;; --- put-down ---
  ;; Success: (obj-at ?o ?l) (clear-hand ?r), delete (holding ?r ?o) (on-stove ?o ?s),
  ;;          (decrease (energy ?r) 0.3)
  (:action put-down
    :parameters (?r - robot ?o - movable)
    :precondition (and
      (holding ?r ?o)
    )
    :effect (and)
  )

  ;; --- put-in ---
  ;; Success: (in ?o ?c) (clear-hand ?r), delete (holding ?r ?o) (obj-at ?o ?l) (on-stove ?o ?s),
  ;;          (decrease (energy ?r) 0.3)
  (:action put-in
    :parameters (?r - robot ?o - movable ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (holding ?r ?o)
      (or (not (openable ?c)) (open ?c))
    )
    :effect (and)
  )

  ;; --- take-out ---
  ;; Success: (holding ?r ?o), delete (in ?o ?c) (clear-hand ?r),
  ;;          (decrease (energy ?r) 0.3)
  (:action take-out
    :parameters (?r - robot ?o - movable ?c - container)
    :precondition (and
      (clear-hand ?r)
      (co-located ?r ?c)
      (in ?o ?c)
      (or (not (openable ?c)) (open ?c))
    )
    :effect (and)
  )

  ;; --- power-on ---
  ;; Success: (powered ?a), (decrease (energy ?r) 0.2)
  (:action power-on
    :parameters (?r - robot ?a - appliance)
    :precondition (and
      (co-located ?r ?a)
      (not (powered ?a))
    )
    :effect (and)
  )

  ;; --- power-off ---
  ;; Success: delete (powered ?a) (burner-on ?a), (decrease (energy ?r) 0.15)
  (:action power-off
    :parameters (?r - robot ?a - appliance)
    :precondition (and
      (co-located ?r ?a)
      (powered ?a)
    )
    :effect (and)
  )

  ;; --- place-on-stove ---
  ;; Success: (on-stove ?c ?s) (obj-at ?c ?l) (clear-hand ?r), delete (holding ?r ?c),
  ;;          (decrease (energy ?r) 0.2)
  (:action place-on-stove
    :parameters (?r - robot ?c - cookware ?s - stove)
    :precondition (and
      (holding ?r ?c)
      (co-located ?r ?s)
    )
    :effect (and)
  )

  ;; --- ignite-stove ---
  ;; Success: (burner-on ?s), (decrease (energy ?r) 0.2)
  (:action ignite-stove
    :parameters (?r - robot ?s - stove)
    :precondition (and
      (co-located ?r ?s)
      (powered ?s)
      (not (burner-on ?s))
    )
    :effect (and)
  )

  ;; --- extinguish-stove ---
  ;; Success: delete (burner-on ?s), (decrease (energy ?r) 0.1)
  (:action extinguish-stove
    :parameters (?r - robot ?s - stove)
    :precondition (and
      (co-located ?r ?s)
      (burner-on ?s)
    )
    :effect (and)
  )

  ;; --- heat-stove ---
  ;; Success: (increase (surface-temp ?s) 40*?n), (decrease (energy ?r) 0.3)
  (:action heat-stove
    :parameters (?r - robot ?s - stove ?n - number)
    :precondition (and
      (co-located ?r ?s)
      (burner-on ?s)
    )
    :effect (and)
  )

  ;; --- cool-stove ---
  ;; Success: (decrease (surface-temp ?s) 30*?n), (decrease (energy ?r) 0.1)
  (:action cool-stove
    :parameters (?r - robot ?s - stove ?n - number)
    :precondition (and
      (co-located ?r ?s)
      (not (burner-on ?s))
    )
    :effect (and)
  )

  ;; --- heat-on-stove ---
  ;; Success: (increase (water-temp ?c) 20*?n), (decrease (energy ?r) 0.3)
  (:action heat-on-stove
    :parameters (?r - robot ?c - container ?s - stove ?n - number)
    :precondition (and
      (co-located ?r ?s)
      (on-stove ?c ?s)
      (burner-on ?s)
      (hot-surface ?s)
      (has-water ?c)
    )
    :effect (and)
  )

  ;; --- cook-on-stove ---
  ;; Success: (increase (food-temp ?f) 25*?n), (decrease (energy ?r) 0.3)
  (:action cook-on-stove
    :parameters (?r - robot ?f - food ?c - cookware ?s - stove ?n - number)
    :precondition (and
      (co-located ?r ?s)
      (on-stove ?c ?s)
      (burner-on ?s)
      (hot-surface ?s)
      (in ?f ?c)
    )
    :effect (and)
  )

  ;; --- fill-water ---
  ;; Success: (has-water ?k), delete (has-hot-water ?k),
  ;;          (assign (water-temp ?k) 20), (decrease (energy ?r) 0.2)
  (:action fill-water
    :parameters (?r - robot ?k - container ?s - container)
    :precondition (and
      (co-located ?r ?k)
      (co-located ?r ?s)
      (is-sink ?s)
      (or (not (openable ?k)) (open ?k))
    )
    :effect (and)
  )

  ;; --- heat-kettle ---
  ;; Success: (increase (water-temp ?k) 15*?n)
  (:action heat-kettle
    :parameters (?r - robot ?k - appliance ?n - number)
    :precondition (and
      (co-located ?r ?k)
      (powered ?k)
      (has-water ?k)
      (not (open ?k))
    )
    :effect (and)
  )

  ;; --- pour ---
  ;; Success (hot): (has-water ?m) (has-hot-water ?m),
  ;;   delete (has-water ?k) (has-hot-water ?k) (spilled ?m),
  ;;   (assign (water-temp ?m) (water-temp ?k))
  ;; Success (cool): (has-water ?m),
  ;;   delete (has-water ?k) (has-hot-water ?k) (has-hot-water ?m) (spilled ?m),
  ;;   (assign (water-temp ?m) (water-temp ?k))
  ;; Spill: if (pour-in-needs-open ?m) and not (open ?m): water spills
  (:action pour
    :parameters (?r - robot ?k - container ?m - container)
    :precondition (and
      (co-located ?r ?k)
      (co-located ?r ?m)
      (has-water ?k)
      (not (= ?k ?m))
      (or (not (pour-out-needs-open ?k)) (open ?k))
      (or (not (pour-out-needs-closed ?k)) (not (open ?k)))
    )
    :effect (and)
  )

  ;; --- steep-tea ---
  ;; Success: (tea-ready ?m), delete (spilled ?m), (decrease (energy ?r) 0.5)
  (:action steep-tea
    :parameters (?r - robot ?tb - movable ?m - container)
    :precondition (and
      (co-located ?r ?m)
      (in ?tb ?m)
      (>= (water-temp ?m) 80)
    )
    :effect (and)
  )

  ;; --- recharge ---
  ;; Success: (increase (energy ?r) 5)
  (:action recharge
    :parameters (?r - robot ?l - location)
    :precondition (and
      (at ?r ?l)
      (has-charger ?l)
    )
    :effect (and)
  )
)
