def update_teacher(student, teacher, momentum=0.999):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
