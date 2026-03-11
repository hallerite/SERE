(define (problem ferry-4)
  (:domain ferry)
  (:objects
    a b c - location
    c1 c2 - car
  )
  (:init
    (not-eq a b) (not-eq b a)
    (not-eq a c) (not-eq c a)
    (not-eq b c) (not-eq c b)
    (at-ferry a)
    (at c1 a) (at c2 b)
    (empty-ferry)
  )
  (:goal (and (at c1 c) (at c2 c)))
)
