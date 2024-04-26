import pandas as pd
label_list = [
    'artificial intelligence, agents', 
    'artificial intelligence, data mining', 
    'artificial intelligence, expert systems', 
    'artificial intelligence, games and search', 
    'artificial intelligence, knowledge representation', 
    'artificial intelligence, machine learning, case-based', 
    'artificial intelligence, machine learning, genetic algorithms', 
    'artificial intelligence, machine learning, neural networks', 
    'artificial intelligence, machine learning, probabilistic methods', 
    'artificial intelligence, machine learning, reinforcement learning', 
    'artificial intelligence, machine learning, rule learning', 
    'artificial intelligence, machine learning, theory', 
    'artificial intelligence, nlp', 
    'artificial intelligence, planning', 
    'artificial intelligence, robotics', 
    'artificial intelligence, speech', 
    'artificial intelligence, theorem proving', 
    'artificial intelligence, vision and pattern recognition', 
    'data structures, algorithms and theory, computational complexity', 
    'data structures, algorithms and theory, computational geometry', 
    'data structures, algorithms and theory, formal languages', 
    'data structures, algorithms and theory, hashing', 
    'data structures, algorithms and theory, logic', 
    'data structures, algorithms and theory, parallel', 
    'data structures, algorithms and theory, quantum computing', 
    'data structures, algorithms and theory, randomized', 
    'data structures, algorithms and theory, sorting', 
    'databases, concurrency', 
    'databases, deductive', 
    'databases, object oriented', 
    'databases, performance', 
    'databases, query evaluation', 
    'databases, relational', 
    'databases, temporal', 
    'encryption and compression, compression', 
    'encryption and compression, encryption', 
    'encryption and compression, security', 
    'hardware and architecture, distributed architectures', 
    'hardware and architecture, high performance computing', 
    'hardware and architecture, input output and storage', 
    'hardware and architecture, logic design', 
    'hardware and architecture, memory structures', 
    'hardware and architecture, microprogramming', 
    'hardware and architecture, vlsi', 
    'human computer interaction, cooperative', 
    'human computer interaction, graphics and virtual reality', 
    'human computer interaction, interface design', 
    'human computer interaction, multimedia', 
    'human computer interaction, wearable computers', 
    'information retrieval, digital library', 
    'information retrieval, extraction', 
    'information retrieval, filtering', 
    'information retrieval, retrieval', 
    'networking, internet', 
    'networking, protocols', 
    'networking, routing', 
    'networking, wireless', 
    'operating systems, distributed', 
    'operating systems, fault tolerance', 
    'operating systems, memory management', 
    'operating systems, realtime', 
    'programming, compiler design', 
    'programming, debugging', 
    'programming, functional', 
    'programming, garbage collection', 
    'programming, java', 
    'programming, logic', 
    'programming, object oriented', 
    'programming, semantics', 
    'programming, software development']

long_text = """
Artificial Intelligence, Agents: This category involves the study of intelligent agents, which are entities that perceive their environment and act upon it to achieve their goals. Research in this area often focuses on agent architectures, decision-making processes, and interaction with environments.
Artificial Intelligence, Data Mining: Data mining refers to the process of discovering patterns, trends, and insights from large datasets. In this category, researchers explore algorithms and techniques for extracting useful information from data, with applications in areas like business intelligence, marketing, and scientific research.
Artificial Intelligence, Expert Systems: Expert systems are computer programs that mimic the decision-making abilities of human experts in specific domains. Research in this category involves developing knowledge representation and reasoning techniques to build systems capable of solving complex problems in various domains.
Artificial Intelligence, Games and Search: This category encompasses research on algorithms and strategies for playing games and conducting search in large problem spaces. It includes topics such as game theory, adversarial search, and heuristic search algorithms like A*.
Artificial Intelligence, Knowledge Representation: Knowledge representation involves encoding information in a format that can be used by intelligent systems to reason and make decisions. Researchers in this category develop formalisms and languages for representing knowledge, such as semantic networks and ontologies.
Artificial Intelligence, Machine Learning, Case-Based: Case-based machine learning involves learning from past experiences or cases to make decisions or solve new problems. Research in this area focuses on techniques for retrieving and adapting relevant cases to new situations.
Artificial Intelligence, Machine Learning, Genetic Algorithms: Genetic algorithms are optimization techniques inspired by the process of natural selection and evolution. In this category, researchers explore the use of genetic algorithms for solving complex optimization and search problems.
Artificial Intelligence, Machine Learning, Neural Networks: Neural networks are computational models inspired by the structure and function of the human brain. Research in this category involves developing neural network architectures, training algorithms, and applications in areas like image recognition and natural language processing.
Artificial Intelligence, Machine Learning, Probabilistic Methods: Probabilistic methods in machine learning involve modeling uncertainty and making decisions based on probabilistic reasoning. Research in this category includes Bayesian networks, Markov models, and probabilistic graphical models.
Artificial Intelligence, Machine Learning, Reinforcement Learning: Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards. Research in this area focuses on algorithms for learning optimal policies and applications in robotics, gaming, and control systems.
Artificial Intelligence, Machine Learning, Rule Learning: Rule learning involves discovering patterns or rules from data to make predictions or decisions. Research in this category explores algorithms for inducing rule-based models from datasets and their applications in classification, regression, and expert systems.
Artificial Intelligence, Machine Learning, Theory: This category encompasses theoretical research in machine learning, including the analysis of learning algorithms, computational complexity, and the mathematical foundations of learning theory.
Artificial Intelligence, NLP: Natural language processing (NLP) involves the interaction between computers and human language. Research in this category focuses on techniques for understanding, generating, and processing natural language, with applications in machine translation, sentiment analysis, and text summarization.
Artificial Intelligence, Planning: Planning involves generating sequences of actions to achieve desired goals in complex environments. Research in this category explores algorithms and representations for automated planning, scheduling, and decision-making in domains like robotics, logistics, and resource allocation.
Artificial Intelligence, Robotics: Robotics is the interdisciplinary field involving the design, construction, and operation of robots. Research in this category covers topics such as robot perception, motion planning, manipulation, and human-robot interaction.
Artificial Intelligence, Speech: Speech processing involves the analysis and synthesis of spoken language. Research in this category includes speech recognition, speech synthesis, speaker identification, and speech-based interaction systems.
Artificial Intelligence, Theorem Proving: Theorem proving involves the use of formal logic and automated reasoning techniques to prove mathematical theorems or verify the correctness of logical statements. Research in this category focuses on developing algorithms and systems for automated theorem proving and formal verification.
Artificial Intelligence, Vision and Pattern Recognition: Vision and pattern recognition involve the analysis and interpretation of visual data. Research in this category includes topics such as image classification, object detection, scene understanding, and pattern recognition techniques applied to various domains.
Data Structures, Algorithms and Theory, Computational Complexity: Computational complexity theory studies the intrinsic difficulty of computational problems and the resources required to solve them. Research in this category involves analyzing the time and space complexity of algorithms, classifying problems into complexity classes, and investigating the limits of efficient computation.
Data Structures, Algorithms and Theory, Computational Geometry: Computational geometry deals with algorithms and data structures for solving geometric problems. Research in this category includes topics such as geometric algorithms for point location, convex hulls, Voronoi diagrams, and applications in areas like computer graphics, robotics, and geographic information systems.
Data Structures, Algorithms and Theory, Formal Languages: Formal language theory studies the properties of formal languages and their relationship to automata theory and formal grammars. Research in this category involves defining and analyzing various types of formal languages, studying their grammatical rules, and investigating their applications in compiler design, parsing, and text processing.
Data Structures, Algorithms and Theory, Hashing: Hashing is a technique used to map data of arbitrary size to fixed-size values. Research in this category focuses on designing efficient hashing algorithms, analyzing their performance characteristics, and exploring applications in data storage, retrieval, and cryptography.
Data Structures, Algorithms and Theory, Logic: Logic forms the foundation of computer science and mathematics, providing a formal framework for reasoning and computation. Research in this category encompasses various branches of logic, including propositional logic, predicate logic, modal logic, and their applications in automated reasoning, formal verification, and artificial intelligence.
Data Structures, Algorithms and Theory, Parallel: Parallel algorithms and parallel computing involve the simultaneous execution of multiple tasks to solve a computational problem. Research in this category explores techniques for designing efficient parallel algorithms, analyzing their scalability and performance, and addressing challenges related to synchronization, load balancing, and communication in parallel systems.
Data Structures, Algorithms and Theory, Quantum Computing: Quantum computing is an emerging field that explores the use of quantum-mechanical phenomena to perform computation. Research in this category involves developing quantum algorithms, designing quantum circuits, studying quantum complexity classes, and investigating potential applications in cryptography, optimization, and simulation.
Data Structures, Algorithms and Theory, Randomized: Randomized algorithms use randomness to solve computational problems efficiently or with high probability. Research in this category includes the design and analysis of randomized algorithms, their applications in areas like optimization, graph theory, and cryptography, and the study of probabilistic methods in algorithm design.
Data Structures, Algorithms and Theory, Sorting: Sorting algorithms reorder elements in a collection according to a specified order criterion. Research in this category focuses on developing efficient sorting algorithms, analyzing their time and space complexity, and exploring variations such as stable sorting, external sorting, and parallel sorting algorithms.
Databases, Concurrency: Concurrency control in databases involves managing simultaneous access to shared data by multiple transactions to ensure consistency and isolation. Research in this category includes techniques for concurrency control, such as locking, timestamp ordering, and optimistic concurrency control, and their impact on transaction throughput, scalability, and performance.
Databases, Deductive: Deductive databases extend traditional relational databases with logic-based query languages and inference capabilities. Research in this category involves developing deductive database systems, studying query optimization techniques, and exploring applications in knowledge representation, expert systems, and semantic web technologies.
Databases, Object Oriented: Object-oriented databases store data in the form of objects, which encapsulate both data and behavior. Research in this category includes modeling object-oriented data schemas, developing query languages and query optimization techniques, and addressing issues related to object identity, inheritance, and persistence.
Databases, Performance: Database performance optimization involves tuning various components of a database system to improve query execution time, throughput, and resource utilization. Research in this category encompasses techniques for indexing, query optimization, caching, parallel execution, and storage management to enhance database performance under different workloads and system configurations.
Databases, Query Evaluation: Query evaluation in databases involves processing user queries and retrieving relevant data from the underlying database tables. Research in this category focuses on developing efficient query evaluation algorithms, query optimization techniques, and indexing strategies to minimize query response time and improve overall system performance.
Databases, Relational: Relational databases store data in tabular form and support structured query language (SQL) for data manipulation and retrieval. Research in this category includes relational data model design, normalization, transaction management, and query processing and optimization techniques for relational database systems.
Databases, Temporal: Temporal databases store and manage time-varying data, allowing users to query and analyze data as it evolves over time. Research in this category involves temporal data modeling, versioning, transaction-time and valid-time semantics, and temporal query languages for querying historical and future data in temporal database systems.
Encryption and Compression, Compression: Compression algorithms reduce the size of data by encoding it using fewer bits than the original representation. Research in this category includes developing lossless and lossy compression algorithms, analyzing their compression ratios and computational complexity, and exploring applications in data storage, transmission, and multimedia compression.
Encryption and Compression, Encryption: Encryption algorithms transform plaintext data into ciphertext to protect it from unauthorized access or interception. Research in this category encompasses symmetric and asymmetric encryption techniques, cryptographic protocols, key management, and cryptographic primitives for ensuring data confidentiality, integrity, and authenticity.
Encryption and Compression, Security: Security encompasses various measures to protect computer systems, networks, and data from unauthorized access, attacks, and breaches. Research in this category includes cryptography, access control, authentication, intrusion detection, security protocols, and security mechanisms for ensuring confidentiality, integrity, and availability of information.
Hardware and Architecture, Distributed Architectures: Distributed architectures involve the design and implementation of computer systems composed of interconnected nodes that communicate and collaborate to achieve common goals. Research in this category includes distributed computing models, communication protocols, fault tolerance mechanisms, and scalability strategies for building reliable and scalable distributed systems.
Hardware and Architecture, High Performance Computing: High-performance computing (HPC) focuses on designing and building computer systems capable of delivering high computational performance for demanding applications. Research in this category includes parallel processing architectures, memory hierarchies, interconnection networks, and optimization techniques for improving the performance of HPC systems across different workloads and domains.
Hardware and Architecture, Input Output and Storage: Input/output (I/O) and storage systems involve the design and management of devices and interfaces for reading from and writing to external storage media. Research in this category includes disk scheduling algorithms, I/O buffering strategies, storage virtualization, and reliability and performance optimization techniques for I/O and storage subsystems.
Hardware and Architecture, Logic Design: Logic design involves designing and implementing digital circuits and systems using logic gates and electronic components. Research in this category includes logic synthesis, circuit optimization, design automation tools, and hardware description languages for developing efficient and reliable digital systems.
Hardware and Architecture, Memory Structures: Memory structures refer to the organization and management of memory resources in computer systems. Research in this category includes memory hierarchy design, cache coherence protocols, memory management techniques, and optimizations for improving memory performance and efficiency in both traditional and emerging memory technologies.
Hardware and Architecture, Microprogramming: Microprogramming involves designing microprograms that control the behavior of microprocessors and execute machine instructions at a lower level of abstraction. Research in this category includes microarchitecture design, instruction set architecture (ISA) design, and techniques for optimizing microprograms for performance, power efficiency, and compatibility.
Hardware and Architecture, VLSI: Very large-scale integration (VLSI) involves designing and fabricating integrated circuits (ICs) containing millions or billions of transistors on a single chip. Research in this category includes VLSI design methodologies, layout techniques, fabrication technologies, and tools for designing complex digital and analog circuits for various applications.
Human Computer Interaction, Cooperative: Cooperative human-computer interaction (HCI) involves designing systems that support collaboration and communication between humans and computers to achieve shared goals. Research in this category includes collaborative interfaces, groupware, distributed cognition, and social computing technologies for enabling effective teamwork and cooperation in interactive systems.
Human Computer Interaction, Graphics and Virtual Reality: Graphics and virtual reality (VR) technologies enable the creation and manipulation of visual representations in computer systems. Research in this category includes 3D graphics rendering, virtual reality systems, augmented reality applications, human perception in virtual environments, and user interfaces for immersive experiences.
Human Computer Interaction, Interface Design: Interface design involves designing the interaction between users and computer systems through graphical, auditory, and tactile interfaces. Research in this category includes user interface design principles, usability testing methodologies, interaction techniques, and interface prototyping tools for creating intuitive and efficient user interfaces across different platforms and devices.
Human Computer Interaction, Multimedia: Multimedia systems integrate various forms of media, such as text, graphics, audio, video, and animation, to create interactive and engaging user experiences. Research in this category includes multimedia authoring tools, content representation and synchronization, multimedia streaming and compression, and user interfaces for accessing and interacting with multimedia content.
Human Computer Interaction, Wearable Computers: Wearable computers are small, portable devices worn on the body that provide continuous access to computing and communication capabilities. Research in this category includes wearable device design, sensor integration, context-aware computing, human augmentation technologies, and user interfaces for wearable applications in healthcare, fitness, productivity, and entertainment.
Information Retrieval, Digital Library: Digital libraries provide access to large collections of digital resources, such as documents, multimedia, and datasets, organized and searchable through information retrieval systems. Research in this category includes indexing and retrieval techniques, metadata standards, digital preservation, user interfaces for browsing and searching digital collections, and applications of digital libraries in education, research, and cultural heritage.
Information Retrieval, Extraction: Information extraction involves automatically identifying and extracting structured information from unstructured or semi-structured text data. Research in this category includes named entity recognition, relationship extraction, event detection, and sentiment analysis techniques for extracting relevant information from text documents, web pages, social media, and other sources.
Information Retrieval, Filtering: Information filtering involves selecting and presenting relevant information to users based on their preferences, interests, and context. Research in this category includes content-based filtering, collaborative filtering, recommender systems, and personalized search techniques for filtering and ranking information to meet user needs and improve information access and discovery.
Information Retrieval, Retrieval: Information retrieval (IR) involves finding relevant documents or information items in response to user queries. Research in this category includes indexing and searching techniques, ranking algorithms, relevance feedback mechanisms, evaluation methodologies, and user interfaces for effective information retrieval in various domains such as web search, enterprise search, and digital libraries.
Networking, Internet: The Internet is a global network of interconnected computer networks that facilitate communication and information exchange between users and devices worldwide. Research in this category includes Internet protocols, network architectures, routing algorithms, congestion control mechanisms, and applications of Internet technologies in areas like web services, social networking, and cloud computing.
Networking, Protocols: Network protocols define the rules and conventions for communication between devices in a computer network. Research in this category includes protocol design and analysis, protocol implementations, interoperability testing, and standardization efforts for ensuring reliable and efficient communication between heterogeneous networked systems.
Networking, Routing: Routing involves determining the optimal paths for data packets to travel from a source to a destination in a computer network. Research in this category includes routing algorithms, routing protocols, traffic engineering techniques, and quality of service (QoS) mechanisms for improving the efficiency, reliability, and scalability of routing in wired and wireless networks.
Networking, Wireless: Wireless networking enables communication between devices without the need for physical wired connections. Research in this category includes wireless communication technologies, mobile networking protocols, wireless sensor networks, ad hoc network routing algorithms, and wireless network security mechanisms for supporting reliable and efficient communication in diverse wireless environments.
Operating Systems, Distributed: Distributed operating systems manage resources and coordinate the execution of processes across multiple interconnected computers or nodes in a distributed computing environment. Research in this category includes distributed system architectures, process scheduling algorithms, communication protocols, fault tolerance mechanisms, and distributed file systems for building scalable and reliable distributed operating systems.
Operating Systems, Fault Tolerance: Fault tolerance in operating systems involves designing systems that can continue to operate properly in the presence of hardware or software failures. Research in this category includes fault detection and recovery mechanisms, redundancy techniques, checkpointing and rollback strategies, and distributed consensus algorithms for ensuring system reliability and availability.
Operating Systems, Memory Management: Memory management in operating systems involves allocating and deallocating memory resources to processes, managing memory hierarchies, and ensuring efficient memory utilization. Research in this category includes memory allocation algorithms, virtual memory management, memory protection mechanisms, and garbage collection techniques for improving memory performance and reliability.
Operating Systems, Realtime: Real-time operating systems (RTOS) provide predictable and deterministic response times for processing time-critical tasks in embedded systems and other applications with strict timing requirements. Research in this category includes real-time scheduling algorithms, resource reservation mechanisms, deadline enforcement techniques, and temporal partitioning strategies for guaranteeing timely and predictable system behavior.
Programming, Compiler Design: Compiler design involves creating software tools that translate high-level programming languages into machine-readable code for execution on specific hardware platforms. Research in this category includes compiler optimizations, code generation techniques, parsing algorithms, intermediate representations, and compiler verification methods for building efficient and reliable compilers.
Programming, Debugging: Debugging is the process of identifying and fixing errors or bugs in software programs. Research in this category includes debugging tools, techniques for fault localization and diagnosis, program analysis methods, automated debugging approaches, and debugging support in integrated development environments (IDEs) for improving software quality and developer productivity.
Programming, Functional: Functional programming is a programming paradigm that emphasizes the use of functions as first-class entities and immutable data structures. Research in this category includes functional language design, type systems, lambda calculus, higher-order functions, and functional programming techniques for writing concise, modular, and maintainable software.
Programming, Garbage Collection: Garbage collection is a memory management technique that automatically deallocates memory occupied by objects that are no longer in use, reducing the risk of memory leaks and manual memory management errors. Research in this category includes garbage collection algorithms, heap organization strategies, and performance optimizations for efficient memory reclaiming in programming languages and runtime environments.
Programming, Java: Java is a popular programming language known for its platform independence, object-oriented features, and robustness. Research in this category includes language enhancements, runtime optimizations, virtual machine design, concurrency libraries, and Java development tools for improving the performance, scalability, and usability of Java-based software systems.
Programming, Logic: Logic programming is a programming paradigm that uses logical inference rules to specify computation. Research in this category includes logic language semantics, theorem proving, constraint solving, and applications of logic programming in areas such as artificial intelligence, database systems, and declarative problem solving.
Programming, Object Oriented: Object-oriented programming (OOP) is a programming paradigm that organizes software into objects, which encapsulate data and behavior. Research in this category includes object-oriented language design, inheritance and polymorphism mechanisms, design patterns, and object-oriented analysis and design methodologies for building modular, reusable, and maintainable software systems.
Programming, Semantics: Programming language semantics define the meaning of program constructs and their behavior during execution. Research in this category includes formal semantics, type systems, operational semantics, denotational semantics, and program analysis techniques for reasoning about program correctness, optimization, and verification.
Programming, Software Development: Software development encompasses the entire process of designing, implementing, testing, and maintaining software systems. Research in this category includes software engineering methodologies, development tools and environments, version control systems, project management techniques, and best practices for improving software quality, productivity, and collaboration.
"""
label_desc = long_text.strip().split('\n')
f_label_list = []
desc_list = []
for i in range(70):
    label = label_list[i]
    l, desc = label_desc[i].split(': ')[0], label_desc[i].split(': ')[1]
    assert l.lower() == label, f"{l}_{label}"
    f_label_list.append(label)
    desc_list.append(desc)

res = {
    'label': f_label_list,
    'description': desc_list
}
df = pd.DataFrame(res)
df.to_csv('./categories.csv', index=False)
