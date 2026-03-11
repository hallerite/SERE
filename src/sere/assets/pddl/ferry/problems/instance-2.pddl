(define (problem ferry-2)
  (:domain ferry)
  (:objects
    a b - location
    c1 c2 - car
  )
  (:init
    (not-eq a b) (not-eq b a)
    (at-ferry a)
    (at c1 a) (at c2 a)
    (empty-ferry)
  )
  (:goal (and (at c1 b) (at c2 b)))
)
