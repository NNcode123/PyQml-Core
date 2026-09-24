#include <iostream>
#include <vector>
#include "intrusive_ptr.hpp"
#include <thread>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>



#include <iostream>


void storage_deleter(void* p)
{
    std::cout << "Deleting buffer...\n";
    delete[] static_cast<int*>(p);
}


static int storage_destroy_count = 0;





struct TestStorage : public Storage
{
    using Storage::Storage;

    ~TestStorage()
    {
        storage_destroy_count++;
        std::cout << "Storage destructor called\n";
    }
};


void temporary_ownership_churn()
{
    std::cout << "\n=== Temporary Ownership Churn ===\n";

    {
        auto root = make_intrusive<TestStorage, int>(
            new int[100],
            100
        );

        std::vector<pyq_intrusive_ptr<TestStorage>> holders;

        for (int i = 0; i < 100; i++)
        {
            holders.push_back(root);

            if (i % 2 == 0)
            {
                auto temp = holders.back();
                holders.pop_back();

                holders.push_back(std::move(temp));
            }

            if (i % 3 == 0)
            {
                auto copy = root;
            }

            if (i % 10 == 0)
            {
                auto extra = holders;
            }
        }

        holders.clear();
    }

    assert(storage_destroy_count == 1);
}


void vector_reallocation_stress()
{
    std::cout << "\n=== Vector Reallocation Stress ===\n";

    {
        auto root = make_intrusive<TestStorage, int>(
            new int[50],
            50
        );

        std::vector<pyq_intrusive_ptr<TestStorage>> vec;

        for (int i = 0; i < 100; i++)
        {
            vec.push_back(root);
        }

        while (!vec.empty())
            vec.pop_back();
    }

    assert(storage_destroy_count == 2);
}


void move_chain_stress()
{
    std::cout << "\n=== Move Chain Stress ===\n";

    {
        auto a = make_intrusive<TestStorage, int>(
            new int[10],
            10
        );

        auto b = std::move(a);
        auto c = std::move(b);
        auto d = std::move(c);
        auto e = std::move(d);

        a = std::move(e);
    }

    assert(storage_destroy_count == 3);
}


void self_assignment_stress()
{
    std::cout << "\n=== Self Assignment Stress ===\n";

    {
        auto ptr = make_intrusive<TestStorage, int>(
            new int[20],
            20
        );

        ptr = ptr;
        ptr = ptr;
    }

    assert(storage_destroy_count == 4);
}


void copy_after_move_stress()
{
    std::cout << "\n=== Copy After Move Stress ===\n";

    {
        auto original = make_intrusive<TestStorage, int>(
            new int[30],
            30
        );

        auto moved = std::move(original);

        auto copy1 = moved;
        auto copy2 = copy1;
        auto copy3 = copy2;

        original = std::move(moved);
    }

    assert(storage_destroy_count == 5);
}

int main()
{
    std::cout << "Creating Storage...\n";

    pyq_intrusive_ptr<Storage> out(new size_t[20], 20);


    //auto* raw = new int[10];
    //auto* storage = new Storage(raw, storage_deleter);

    std::cout << "\nConstructing p1\n";
    pyq_intrusive_ptr<Storage> p1 = make_intrusive<Storage>(new int[10], storage_deleter, 10);

    {
        std::cout << "\nCopy constructing p2 from p1\n";
        pyq_intrusive_ptr<Storage> p2(p1);

        std::cout << "\nMove constructing p3 from p2\n";
        pyq_intrusive_ptr<Storage> p3(std::move(p2));

        std::cout << "\nDefault constructing p4\n";
        pyq_intrusive_ptr<Storage> p4{};

        std::cout << "\nCopy assigning p4 = p3\n";
        p4 = p3;

        std::cout << "\nMove assigning p3 = std::move(p1)\n";
        p3 = std::move(p1);

        std::cout << "\nLeaving inner scope...\n";
    }

     std::cout << "\n=== Repeated Copies ===\n";
    {
        auto p2 = p1;
        auto p3 = p2;
        auto p4 = p3;
        auto p5 = p4;
    }

    // ------------------------------------------------------------
    // Assignment chain
    // ------------------------------------------------------------
    std::cout << "\n=== Assignment Chain ===\n";
    {
        pyq_intrusive_ptr<Storage> a(new Storage(new int[5], storage_deleter, 5));
        pyq_intrusive_ptr<Storage> b(new Storage(new int[5], storage_deleter, 5));
        pyq_intrusive_ptr<Storage> c(new Storage(new int[5], storage_deleter, 5));

        a = b;
        b = c;
        c = a;
    }

    // ------------------------------------------------------------
    // Self assignment
    // ------------------------------------------------------------
    std::cout << "\n=== Self Assignment ===\n";
    {
        p1 = p1;
    }

    // ------------------------------------------------------------
    // Move chain
    // ------------------------------------------------------------
    std::cout << "\n=== Move Chain ===\n";
    {
        auto m1 = std::move(p1);
        auto m2 = std::move(m1);
        auto m3 = std::move(m2);

        p1 = std::move(m3);
    }

    // ------------------------------------------------------------
    // Copy after move
    // ------------------------------------------------------------
    std::cout << "\n=== Copy After Move ===\n";
    {
        auto moved = std::move(p1);
        auto copy = moved;

        p1 = std::move(moved);
    }

    // ------------------------------------------------------------
    // Scope nesting
    // ------------------------------------------------------------
    std::cout << "\n=== Nested Scope ===\n";
    {
        auto a = p1;

        {
            auto b = a;

            {
                auto c = b;
            }

            auto d = a;
        }
    }

    // ------------------------------------------------------------
    // Many copies
    // ------------------------------------------------------------
    std::cout << "\n=== Vector Stress Test ===\n";
    {
        std::vector<pyq_intrusive_ptr<Storage>> vec;

        for (int i = 0; i < 1000; ++i)
        {
            vec.push_back(p1);
        }

        vec.clear();
    }

    std::cout << "\n=== End of main ===\n";

    std::cout << "\nProgram ending...\n";

     temporary_ownership_churn();

    vector_reallocation_stress();

    move_chain_stress();

    self_assignment_stress();

    copy_after_move_stress();


    std::cout 
        << "\nAll intrusive_ptr tests passed!\n";

    std::cout 
        << "Destructor count = "
        << storage_destroy_count
        << "\n";
}






