(define (problem ferry-1)
  (:domain ferry)
  (:objects
    a b - location
    c1 - car
  )
  (:init
    (not-eq a b) (not-eq b a)
    (at-ferry a)
    (at c1 a)
    (empty-ferry)
  )
  (:goal (at c1 b))
)
