(define (problem ferry-7)
  (:domain ferry)
  (:objects
    a b c d - location
    c1 c2 c3 c4 c5 - car
  )
  (:init
    (not-eq a b) (not-eq b a) (not-eq a c) (not-eq c a)
    (not-eq a d) (not-eq d a) (not-eq b c) (not-eq c b)
    (not-eq b d) (not-eq d b) (not-eq c d) (not-eq d c)
    (at-ferry a)
    (at c1 a) (at c2 a) (at c3 b) (at c4 c) (at c5 d)
    (empty-ferry)
  )
  (:goal (and (at c1 d) (at c2 c) (at c3 a) (at c4 b) (at c5 a)))
)
