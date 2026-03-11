;; assembly domain – PDDL encoding
;; auto-generated from assembly.yaml

(define (domain assembly)
  (:requirements :strips :typing :fluents :derived-predicates
                 :negative-preconditions :equality)

  ;; ---------- types ----------
  (:types
    robot location object number
    movable container assembly machine - object
    tool part - movable
  )

  ;; ---------- predicates ----------
  (:predicates
    ;; spatial / kinematic
    (at ?r - robot ?l - location)
    (obj-at ?o - object ?l - location)
    (co-located ?x - object ?y - object)
    (adjacent ?l1 - location ?l2 - location)

    ;; grasping / hand state
    (holding ?r - robot ?o - object)
    (clear-hand ?r - robot)

    ;; containers & openness
    (in ?o - object ?c - container)
    (open ?c - container)

    ;; tools & equipment state
    (equipped ?r - robot ?t - tool)
    (tool-for ?t - tool ?op - object)

    ;; assembly relations
    (aligned ?p - part ?a - assembly)
    (installed ?p - part ?a - assembly)
    (fastened ?p - part ?a - assembly)
    (damaged ?o - object)

    ;; infrastructure / facilities
    (has-charger ?l - location)
    (has-fixture ?l - location)

    ;; machines
    (powered ?m - machine)

    ;; quality of outcome
    (defective ?a - assembly)
    (needs-rework ?a - assembly)
  )

  ;; ---------- functions / fluents ----------
  (:functions
    (battery-cap ?r - robot)
    (energy ?r - robot)
    (torque ?t - tool)
    (quality ?a - assembly)
  )

  ;; ---------- derived predicates (co-located rules) ----------
  ;; rule 1: robot-robot via (at)
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (at ?x ?l) (at ?y ?l))))

  ;; rule 2: robot-object via (at) and (obj-at)
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (at ?x ?l) (obj-at ?y ?l))))

  ;; rule 3: object-robot via (obj-at) and (at)
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (obj-at ?x ?l) (at ?y ?l))))

  ;; rule 4: object-object via (obj-at)
  (:derived (co-located ?x - object ?y - object)
    (exists (?l - location)
      (and (obj-at ?x ?l) (obj-at ?y ?l))))

  ;; rule 5: x in container, container co-located with y
  (:derived (co-located ?x - object ?y - object)
    (exists (?c - container)
      (and (in ?x ?c) (co-located ?c ?y))))

  ;; rule 6: y in container, x co-located with container
  (:derived (co-located ?x - object ?y - object)
    (exists (?c - container)
      (and (in ?y ?c) (co-located ?x ?c))))

  ;; rule 7: robot holding x, robot co-located with y
  (:derived (co-located ?x - object ?y - object)
    (exists (?r - robot)
      (and (holding ?r ?x) (co-located ?r ?y))))

  ;; rule 8: robot holding y, x co-located with robot
  (:derived (co-located ?x - object ?y - object)
    (exists (?r - robot)
      (and (holding ?r ?y) (co-located ?x ?r))))

  ;; ---------- actions ----------

  ;; wait – do nothing for n ticks
  (:action wait
    :parameters (?n - number)
    :precondition (and)
    :effect (and)
    ;; success: no state change
  )

  ;; move – navigate between adjacent locations
  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
      (adjacent ?from ?to)
      (>= (energy ?r) 1)
      (not (= ?from ?to))
    )
    :effect (and)
    ;; success: (not (at ?r ?from)) (at ?r ?to) (decrease (energy ?r) 1)
  )

  ;; open – open a container
  (:action open
    :parameters (?r - robot ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (not (open ?c))
    )
    :effect (and)
    ;; success: (open ?c) (decrease (energy ?r) 0.2)
  )

  ;; close – close a container
  (:action close
    :parameters (?r - robot ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (open ?c)
    )
    :effect (and)
    ;; success: (not (open ?c)) (decrease (energy ?r) 0.2)
  )

  ;; pick-up – grasp a movable object
  (:action pick-up
    :parameters (?r - robot ?o - movable)
    :precondition (and
      (co-located ?r ?o)
      (clear-hand ?r)
    )
    :effect (and)
    ;; success: (holding ?r ?o) (not (obj-at ?o ?l)) (not (clear-hand ?r))
    ;;          (decrease (energy ?r) 0.3)
  )

  ;; put-down – release a held movable object
  (:action put-down
    :parameters (?r - robot ?o - movable)
    :precondition (and
      (holding ?r ?o)
    )
    :effect (and)
    ;; success: (obj-at ?o ?l) (clear-hand ?r) (not (holding ?r ?o))
    ;;          (decrease (energy ?r) 0.3)
  )

  ;; put-in – place a movable into an open container
  (:action put-in
    :parameters (?r - robot ?o - movable ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (holding ?r ?o)
      (open ?c)
    )
    :effect (and)
    ;; success: (in ?o ?c) (clear-hand ?r) (not (holding ?r ?o))
    ;;          (not (obj-at ?o ?l)) (decrease (energy ?r) 0.3)
  )

  ;; take-out – remove a movable from an open container
  (:action take-out
    :parameters (?r - robot ?o - movable ?c - container)
    :precondition (and
      (co-located ?r ?c)
      (in ?o ?c)
      (open ?c)
      (clear-hand ?r)
    )
    :effect (and)
    ;; success: (holding ?r ?o) (not (in ?o ?c)) (not (clear-hand ?r))
    ;;          (decrease (energy ?r) 0.3)
  )

  ;; equip-tool – equip a tool currently held in hand
  (:action equip-tool
    :parameters (?r - robot ?t - tool)
    :precondition (and
      (holding ?r ?t)
    )
    :effect (and)
    ;; success: (equipped ?r ?t) (clear-hand ?r) (not (holding ?r ?t))
    ;;          (decrease (energy ?r) 0.2)
  )

  ;; unequip-tool – remove equipped tool back to hand
  (:action unequip-tool
    :parameters (?r - robot ?t - tool)
    :precondition (and
      (equipped ?r ?t)
      (clear-hand ?r)
    )
    :effect (and)
    ;; success: (holding ?r ?t) (not (equipped ?r ?t)) (not (clear-hand ?r))
    ;;          (decrease (energy ?r) 0.2)
  )

  ;; align – align a part with an assembly
  (:action align
    :parameters (?r - robot ?p - part ?a - assembly)
    :precondition (and
      (co-located ?r ?p)
      (co-located ?r ?a)
    )
    :effect (and)
    ;; success: (aligned ?p ?a) (decrease (energy ?r) 0.5)
  )

  ;; fasten – fasten a part to an assembly using an equipped tool
  ;; note: base add effects always applied; conditional block overrides
  ;;       (installed ?p ?a) when the part is damaged
  (:action fasten
    :parameters (?r - robot ?p - part ?a - assembly ?t - tool)
    :precondition (and
      (equipped ?r ?t)
      (tool-for ?t ?a)
      (aligned ?p ?a)
      (co-located ?r ?p)
      (co-located ?r ?a)
    )
    :effect (and
      (fastened ?p ?a)
      (installed ?p ?a)
      (when (damaged ?p)
        (not (installed ?p ?a)))
    )
    ;; outcomes handled by SERE runtime:
    ;; hard_defect: (defective ?a) (assign (quality ?a) 0.20) (decrease (energy ?r) 1.0)
    ;; soft_defect: (needs-rework ?a) (assign (quality ?a) 0.65) (decrease (energy ?r) 1.0)
    ;; success: (not (defective ?a)) (not (needs-rework ?a))
    ;;          (assign (quality ?a) 0.95) (decrease (energy ?r) 1.0)
  )

  ;; set-torque – adjust the torque on an equipped tool
  (:action set-torque
    :parameters (?r - robot ?t - tool ?lvl - object)
    :precondition (and
      (equipped ?r ?t)
    )
    :effect (and)
    ;; success: (decrease (energy ?r) 0.1)
  )

  ;; unfasten – remove a fastened part from an assembly
  (:action unfasten
    :parameters (?r - robot ?p - part ?a - assembly ?t - tool)
    :precondition (and
      (equipped ?r ?t)
      (tool-for ?t ?a)
      (fastened ?p ?a)
      (co-located ?r ?p)
      (co-located ?r ?a)
      (clear-hand ?r)
    )
    :effect (and)
    ;; success: (holding ?r ?p) (not (fastened ?p ?a)) (not (installed ?p ?a))
    ;;          (not (clear-hand ?r)) (not (obj-at ?p ?l))
    ;;          (decrease (energy ?r) 1.0)
  )

  ;; place-on-bench – put a movable down on the workbench
  (:action place-on-bench
    :parameters (?r - robot ?o - movable)
    :precondition (and
      (holding ?r ?o)
    )
    :effect (and)
    ;; success: (obj-at ?o ?l) (clear-hand ?r) (not (holding ?r ?o))
    ;;          (decrease (energy ?r) 0.3)
  )

  ;; power-on – turn on a machine
  (:action power-on
    :parameters (?r - robot ?m - machine)
    :precondition (and
      (co-located ?r ?m)
      (not (powered ?m))
    )
    :effect (and)
    ;; success: (powered ?m) (decrease (energy ?r) 0.2)
  )

  ;; recharge – recharge at a location with a charger
  (:action recharge
    :parameters (?r - robot ?l - location)
    :precondition (and
      (at ?r ?l)
      (has-charger ?l)
    )
    :effect (and)
    ;; success: (increase (energy ?r) 5)
  )

  ;; qc-check – quality-control inspection of an assembly
  (:action qc-check
    :parameters (?r - robot ?a - assembly)
    :precondition (and
      (co-located ?r ?a)
    )
    :effect (and)
    ;; success: observation only, no state change
  )

  ;; realign – re-align a part with an assembly
  (:action realign
    :parameters (?r - robot ?p - part ?a - assembly)
    :precondition (and
      (co-located ?r ?p)
      (co-located ?r ?a)
    )
    :effect (and)
    ;; success: (aligned ?p ?a) (decrease (energy ?r) 0.3)
  )

)
