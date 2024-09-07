# CS186-L15: DB Design: Entity-Relationship Models


## Overview
how to design a database instead of DBMS! :thinking:
![alt text](image.png)

this class mainly focuses on the conceptual design

## Data Models and Relational Levels of Abstraction
### def in Data model
![alt text](image-1.png)
### abstractions
![alt text](image-2.png)
example
![alt text](image-3.png)

## Data Independence
![alt text](image-4.png)

## Entity-Relationship Model (ERM)
![alt text](image-5.png)
### Def
- **Entity**: a real-world object that can be described and identified by a set of attributes
- **Entity Set**: a collection of similar entities
  - all entities in an entity set have the same attributes
  - each entity set has a key
  - each attribute has a domain
- **Relationship**: a connection between two entity sets
![alt text](image-6.png)

## Key and Participation Constraints
### Key Constraints
![alt text](image-7.png)
### Participation Constraints
![alt text](image-8.png)

## Weak Entity 
![alt text](image-9.png)

## Alternative Notation and Terminology
![alt text](image-10.png)
decoder graph :sweat_smile:
![alt text](image-11.png)
math notation :sweat_smile:
![alt text](image-12.png)

## Binary vs Ternary Relationships
![alt text](image-000.png)
上面的更加紧密并且可以记录qty

## Aggregation and Ternary Relationships
![alt text](image-111.png)

## Entities vs Attributes
**Remember**: 
- attributes can not have nested attributes (*atomic attributes* only), if you want to represent nested attributes, use entities instead :thinking:
- entity or attribute? depends on the context!

## Entities vs Relationships
必要时拆出来新的实体来构建新的关系

## Converting ER to Relational
- Entity Set: table
- **many-to-many** Relationship Set: 
  - keys for participating entities, forming a **superkey** for the relation
  - all other attributes
- Key Constraints: think carefully about the uniqueness of the primary key!
- Participation Constraints: usually using NOT NULL
  - ![alt text](image-333.png)
- Weak Entity Set: 
  - ![alt text](image-444.png)
