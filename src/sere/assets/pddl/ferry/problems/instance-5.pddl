(define (problem ferry-5)
  (:domain ferry)
  (:objects
    a b c - location
    c1 c2 c3 c4 - car
  )
  (:init
    (not-eq a b) (not-eq b a)
    (not-eq a c) (not-eq c a)
    (not-eq b c) (not-eq c b)
    (at-ferry a)
    (at c1 a) (at c2 a) (at c3 b) (at c4 b)
    (empty-ferry)
  )
  (:goal (and (at c1 c) (at c2 c) (at c3 a) (at c4 a)))
)
