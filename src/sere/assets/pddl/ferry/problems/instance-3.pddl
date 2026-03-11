(define (problem ferry-3)
  (:domain ferry)
  (:objects
    a b - location
    c1 c2 c3 - car
  )
  (:init
    (not-eq a b) (not-eq b a)
    (at-ferry a)
    (at c1 a) (at c2 a) (at c3 a)
    (empty-ferry)
  )
  (:goal (and (at c1 b) (at c2 b) (at c3 b)))
)
