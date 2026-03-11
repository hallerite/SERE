(define (problem ferry-10)
  (:domain ferry)
  (:objects
    a b c d e f - location
    c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 - car
  )
  (:init
    (not-eq a b) (not-eq b a) (not-eq a c) (not-eq c a)
    (not-eq a d) (not-eq d a) (not-eq a e) (not-eq e a)
    (not-eq a f) (not-eq f a) (not-eq b c) (not-eq c b)
    (not-eq b d) (not-eq d b) (not-eq b e) (not-eq e b)
    (not-eq b f) (not-eq f b) (not-eq c d) (not-eq d c)
    (not-eq c e) (not-eq e c) (not-eq c f) (not-eq f c)
    (not-eq d e) (not-eq e d) (not-eq d f) (not-eq f d)
    (not-eq e f) (not-eq f e)
    (at-ferry a)
    (at c1 a) (at c2 a) (at c3 b) (at c4 b) (at c5 c)
    (at c6 c) (at c7 d) (at c8 d) (at c9 e) (at c10 f)
    (empty-ferry)
  )
  (:goal (and (at c1 f) (at c2 e) (at c3 d) (at c4 c) (at c5 b)
              (at c6 a) (at c7 f) (at c8 e) (at c9 d) (at c10 c)))
)
